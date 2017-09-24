#include <cstdio>
#include <cuda.h>
#include <curand.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "stream_compaction\sharedandbank.h" 
#include "stream_compaction\radix.h" 

#define ERRORCHECK 1

//https://thrust.github.io/doc/group__stream__compaction.html#ga5fa8f86717696de88ab484410b43829b
//https://stackoverflow.com/questions/34103410/glmvec3-and-epsilon-comparison
struct is_dead //needed for thrust's predicate, the last arg in remove_if
{
  __host__ __device__
  bool operator()(const PathSegment& path)
  {
	  //bool colorIsNotBlack = !glm::all(glm::lessThan(glm::abs(path.color), glm::vec3(EPSILON)));
	  //return (!isBlack(path.color) && path.remainingBounces > 0);//we set bounces to zero if color is black in the shader
	  return (path.remainingBounces > 0);
  }
};

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
	//int x = resolution.x-1 - (blockIdx.x*blockDim.x + threadIdx.x);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
//need something for lights
static int numlights = 0;
static int* dev_geomLightIDs = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	//check which geoms are emissive and save their IDs for dev_geomLightIDs later
	std::vector<int> geomLightIDs;
	for (int i = 0; i < scene->geoms.size(); ++i) {
		int materialID = scene->geoms[i].materialid;
		if (scene->materials[materialID].emittance > 0) {
			geomLightIDs.push_back(i);
			numlights++;
		}
	}
	cudaMalloc((void**)&dev_geomLightIDs, numlights * sizeof(int));
  	cudaMemcpy(dev_geomLightIDs, geomLightIDs.data(), numlights * sizeof(int), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_geomLightIDs);
	numlights = 0;

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(const Camera cam, const int iter, const int traceDepth, PathSegment* pathSegments) {
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) return;

	int index = x + (y * cam.resolution.x);
	PathSegment& segment = pathSegments[index];

	segment.ray.origin = cam.position;
	segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

	// TODO: implement antialiasing by jittering the ray
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u(-0.5f, 0.5f);

	//camera top right is (0,0)?
	segment.ray.direction = glm::normalize(cam.view
		//- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
		//- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		- cam.right * cam.pixelLength.x * (x - cam.resolution.x * 0.5f + u(rng))
		- cam.up    * cam.pixelLength.y * (y - cam.resolution.y * 0.5f + u(rng))
	);

	segment.MSPaintPixel = glm::ivec2(cam.resolution.x - 1 - x, y);
	segment.pixelIndex = index;
	segment.remainingBounces = traceDepth;
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++) {
			Geom & geom = geoms[i];

			if (geom.type == CUBE) {
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			} else if (geom.type == SPHERE) {
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;
		} else {
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterialNaive (
  const int iter
  , const int num_paths
	, const ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, const Material * materials
	)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) { return; }

	PathSegment& path = pathSegments[idx];
	const ShadeableIntersection& isect = shadeableIntersections[idx];

	//Hit Nothing
	if (0.f >= isect.t) {// Lots of renderers use 4 channel color, RGBA, where A = alpha, often // used for opacity, in which case they can indicate "no opacity".  // This can be useful for post-processing and image compositing.
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
		return;
	}

	const Material& material = materials[isect.materialId];

	//Hit Light
	if (0.f < material.emittance) {
		path.color *= (material.color * material.emittance); 
		path.remainingBounces = 0;
		return;
	}

	//this was last chance to hit light
	if (0 >= --path.remainingBounces) {
		path.color = glm::vec3(0.f);
		return;
	}

	//Hit Material, generate new ray for the path(wi), get pdf and color for the material, use those to mix with the path's existing color
	//const float pixelX = path.MSPaintPixel.x;
	//const float pixelY = path.MSPaintPixel.y;
	float bxdfPDF; glm::vec3 bxdfColor;
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);
	scatterRayNaive(path, isect, material, bxdfPDF, bxdfColor, rng);
	if (isBlack(bxdfColor)) {//for total internal reflection
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
		return;
	}
	if (bxdfPDF != 0.f) { bxdfColor /= bxdfPDF; }
	path.color *= bxdfColor * absDot(isect.surfaceNormal, path.ray.direction);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(const int nPaths, const glm::ivec2 resolution, glm::vec3 * image, const PathSegment * iterationPaths)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);//8 by 8 pixels
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

	//dev_path holds them all (they are calculated for world space)
	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks, prob only needs to be num_paths not pixelcount
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		shadeMaterialNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);

		//Do compact and determine how many paths are still left, our compact only operates on ints...so use thrust
		//PathSegment *end = thrust::remove_if(dev_paths, dev_paths + num_paths, is_dead());
		//remove_if not working, try partition
		PathSegment *compactend = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_dead());
		num_paths = compactend - dev_paths;
		if (num_paths == 0) {
			iterationComplete = true;
		}
	}

	// Assemble this iteration and apply it to the image
	num_paths = dev_path_end - dev_paths;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, cam.resolution, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
