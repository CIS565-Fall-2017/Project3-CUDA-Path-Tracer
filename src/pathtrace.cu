#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <stream_compaction/efficient.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


#define ERRORCHECK 1

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
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

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
// ...
static int* dev_bool = NULL;//bool array for ray stream compaction
static int* dev_indices = NULL;//indices array for ray stream compaction
static PathSegment* dev_paths_helper = NULL;
static PathSegment* dev_paths_finished = NULL;

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
	cudaMalloc(&dev_indices, pow(2, ilog2ceil(pixelcount)) * sizeof(int));
	cudaMalloc(&dev_bool, pow(2, ilog2ceil(pixelcount)) * sizeof(int));
	cudaMalloc(&dev_paths_helper, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths_finished, pixelcount * sizeof(PathSegment));
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_indices);
	cudaFree(dev_bool);
	cudaFree(dev_paths_helper);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0.f, 1.0f);
		x += u01(rng);
		y += u01(rng);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
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

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect; //World Space Intersection
		glm::vec3 tmp_normal; //World Space Normal

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
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
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.0f) {
			pathSegments[idx].color *= (materialColor * material.emittance);
		}
		// Otherwise, do some pseudo-lighting computation. This is actually more
		// like what you would expect from shading in a rasterizer like OpenGL.
		// TODO: replace this! you should be able to start with basically a one-liner
		else {
			float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
			pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
			pathSegments[idx].color *= u01(rng); // apply some noise because why not
		}
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
		} else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

//Added by Yuxin
__global__ void kernBsdfShader(
	int iter
	, int num_paths, int depth
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
	ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			PathSegment pathSegment = pathSegments[idx];
			Material m = materials[intersection.materialId];
			glm::vec3 intersect = pathSegment.ray.origin + pathSegment.ray.direction*intersection.t;
			scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, m, rng);
		}
		else {
			//no intersection
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void kernMapRayToBoolean(int n, int *bools, const PathSegment *idata) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		if (idata[index].remainingBounces <= 0) {
			bools[index] = 0;
		}
		else {
			bools[index] = 1;
		}
	}
}

__global__ void kernScatterRay(int n, PathSegment *odata,
	const PathSegment *idata, const int *bools, const int *indices) {
	// TODO
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		if (bools[index] == 1) {
			odata[indices[index]] = idata[index];
		}
	}
}

__global__ void kernCalculateCompactResult(int numPath, const int *bools, int *indices) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numPath) {
		//if (index == numPath - 1) {
			indices[index] += bools[index];
		//}
	}
}

__global__ void kernCopyFinishedPath(int numPath, PathSegment* oPath, const PathSegment* iPath) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numPath) {
		if (iPath[index].remainingBounces <= 0) {
			oPath[iPath[index].pixelIndex] = iPath[index];
		}		
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
    const dim3 blockSize2d(8, 8);
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

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0; //Question: do we use this to terminate the path tracing?
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths; //Questions: why not using pixelcount as num_paths???
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	cudaMemset(dev_paths_finished, 0, pixelcount * sizeof(PathSegment));
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		// clean stream compaction helper chunks
		cudaMemset(dev_paths_helper, 0, pixelcount * sizeof(PathSegment));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
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
		kernBsdfShader << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			depth,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		cudaDeviceSynchronize();
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		/*shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
		);*/
		
		//****************Copy current path Segment status to dev_paths_finished************//
		kernCopyFinishedPath << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths_finished, dev_paths);
		cudaDeviceSynchronize();
		//****************Copy current path Segment status to dev_paths_finished************//
		
		//****************Stream Compaction, remove the terminated ray***********************//
		cudaMemset(dev_bool, num_paths, 0);
		cudaMemset(dev_indices, num_paths, 0);
		bool isPowerOfTwo = (num_paths != 0) && ((num_paths & (num_paths - 1)) == 0);
		int npoweroftwo = num_paths;
		if (!isPowerOfTwo) {
			npoweroftwo = pow(2, ilog2ceil(num_paths));
		}
		kernMapRayToBoolean << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_bool, dev_paths);
		cudaDeviceSynchronize();
		//****************Stream Compaction Thrust******************//
		/*thrust::device_ptr<bool> boolFlag(dev_bool);
		thrust::device_ptr<PathSegment> path(dev_paths);
		thrust::remove_if(path, path + num_paths, boolFlag, thrust::logical_not<bool>());
		num_paths = thrust::count_if(boolFlag, boolFlag + num_paths, thrust::identity<bool>());*/
		//****************Stream Compaction Thrust******************//

		cudaMemcpy(dev_indices, dev_bool, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
		//Perform an exclusive sum scan on ibools to get the final indices array
		StreamCompaction::Efficient::gpuEfficientScan(num_paths, npoweroftwo, dev_indices);
		cudaDeviceSynchronize();
		kernScatterRay << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths_helper, dev_paths, dev_bool, dev_indices);
		cudaDeviceSynchronize();

		kernCalculateCompactResult << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_bool, dev_indices);
		cudaDeviceSynchronize();
		int* indices = (int*)malloc(num_paths * sizeof(int));
		cudaMemcpy(indices, dev_indices, num_paths * sizeof(int), cudaMemcpyDeviceToHost);
		int compactResult = indices[num_paths-1];
		delete indices;
		num_paths = compactResult;
		cudaMemset(dev_paths, 0, pixelcount * sizeof(PathSegment));
		cudaMemcpy(dev_paths, dev_paths_helper, compactResult * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		//****************Stream Compaction, remove the terminated ray***********************//		
		
		//If after stream compaction, no rays left in the pool, then set iterationComplete to true
		//iterationComplete = true; 
		// TODO: should be based off stream compaction results.
		if (num_paths < 1) {
		//Debug Purpose
		//if(depth==2){
			//Copy back the result from dev_paths_finished to dev_paths
			cudaMemcpy(dev_paths, dev_paths_finished, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
			iterationComplete = true;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	num_paths = pixelcount;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
