#include <cstdio>
#include <cuda.h>
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

// Include files for stream compaction of unnanted ray paths (Own implementation)
#include "stream_compaction\common.h"
#include "stream_compaction\efficient.h"

// Include files for stream compaction of unwanted ray paths (Thrust implementation)
#include <thrust/partition.h>

// Toggle options for features 1 - ON & 0 - OFF
#define ERRORCHECK	1
#define AA	0
#define DOF	0
#define PATHCOMPACTION	0
#define NAIVEDIRECTLIGHTING	0		// This implementation performs DL on the first Bounce and continues to the next iteration
#define LASTBOUNCEDIRECTLIGHTING  1
#define CACHEFIRSTBOUNCE 0
#define SORTPATHSBYMATERIAL 0
// Toggle for performance analysis
#define PERITERATIONTIMER 0
#define	PERDEPTHTIMER 0

// Enums used for Material Sorting
enum MaterialType { NO, DIFFUSE, REFLECTIVE, REFRACTIVE, LIGHT };

//#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
//#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
//void checkCUDAErrorFn(const char *msg, const char *file, int line) {
//#if ERRORCHECK
//    cudaDeviceSynchronize();
//    cudaError_t err = cudaGetLastError();
//    if (cudaSuccess == err) {
//        return;
//    }
//
//    fprintf(stderr, "CUDA error");
//    if (file) {
//        fprintf(stderr, " (%s:%d)", file, line);
//    }
//    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
//#  ifdef _WIN32
//    getchar();
//#  endif
//    exit(EXIT_FAILURE);
//#endif
//}

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

// Array of indices used for stream compaction
static int* dev_indixes = NULL;
static int* dev_light_indixes = NULL;
static int no_of_lights;

// Array used to cache first bounce
static ShadeableIntersection * dev_intersectionsCache = NULL;

// Array used for sorting the path and the intersection arrays by the material
static int* dev_MaterialSortPath = NULL;
static int* dev_MaterialSortIntersections = NULL;

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
	cudaMalloc(&dev_indixes, pixelcount * sizeof(int));
	cudaMemset(dev_indixes, 0, pixelcount * sizeof(int));

	cudaMalloc(&dev_light_indixes, scene->lightGeometryIndex.size() * sizeof(int));
	cudaMemcpy(dev_light_indixes, scene->lightGeometryIndex.data(), scene->lightGeometryIndex.size() * sizeof(int), cudaMemcpyHostToDevice);

#if CACHEFIRSTBOUNCE
	cudaMalloc(&dev_intersectionsCache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersectionsCache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if SORTPATHSBYMATERIAL
	cudaMalloc(&dev_MaterialSortPath, pixelcount * sizeof(int));
	cudaMemset(dev_MaterialSortPath, -1, pixelcount * sizeof(int));
	cudaMalloc(&dev_MaterialSortIntersections, pixelcount * sizeof(int));
	cudaMemset(dev_MaterialSortIntersections, -1, pixelcount * sizeof(int));
#endif

	// Initializing the number of lights variable
	no_of_lights = scene->lightGeometryIndex.size();

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_indixes);
	cudaFree(dev_light_indixes);
#if CACHEFIRSTBOUNCE
	cudaFree(dev_intersectionsCache);
#endif

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int* indixes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);

		// Adding Antialiasing Jitter on the x and y positions
#if AA
			thrust::uniform_real_distribution<float> u01(0, 1);
			x += u01(rng);
			y += u01(rng);
#endif

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
	
		// Applying Depth of field
#if DOF
			depthOfField(cam, rng, segment);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		// Storing the path segments indixes for compaction later
		//if (PATHCOMPACTION) {
		//	indixes[index] = index;
		//}
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
	, int* indixes
	, int* sortMaterialPaths
	, int* sortmaterialIntersections
	, Material * materials
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index < num_paths)
	{
		int temp_idx = path_index;
		//if (PATHCOMPACTION) {
		//	path_index = indixes[path_index];
		//}

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

#if SORTBYMATERIAL
			sortMaterialPaths[path_index] = NO;
			sortmaterialIntersections[path_index] = NO;
#endif
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].outside = outside;

#if SORTPATHSBYMATERIAL
			Material material = materials[geoms[hit_geom_index].materialid];
			if (material.hasReflective) {
				sortMaterialPaths[path_index] = REFLECTIVE;
				sortmaterialIntersections[path_index] = REFLECTIVE;
			}
			else if (material.hasRefractive) {
				sortMaterialPaths[path_index] = REFRACTIVE;
				sortmaterialIntersections[path_index] = REFRACTIVE;
			}
			else if (material.emittance > 0.f) {
				sortMaterialPaths[path_index] = LIGHT;
				sortmaterialIntersections[path_index] = LIGHT;
			}
			else {
				sortMaterialPaths[path_index] = DIFFUSE;
				sortmaterialIntersections[path_index] = DIFFUSE;
			}
#endif
		}
	}
}

/** Same as above but returns the distance of the point of interscetion along the ray direction
*/

__device__ float findIntersectionsWithScene(Ray ray, int geoms_size, Geom* geoms) {
	float t;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}

		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
		}
	}

	if (hit_geom_index == -1)
	{
		return -1.0f;
	}
	else
	{
		//The ray hits something
		return t_min;
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
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

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

// Shading based on proper BRDF of the materials

__global__ void kernBRDFBasedShader(int iter, int num_paths, ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials, int depth, int* indixes, Geom* geoms, int geoms_size, int* light_indixes, int no_of_lights) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
#if (!PATHCOMPACTION)
		if (pathSegments[idx].remainingBounces <= 0) return;
#endif
	if (idx < num_paths)
	{
		int temp_idx = idx;
		//if (PATHCOMPACTION) {
		//	idx = indixes[idx];
		//}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			PathSegment& tempPS = pathSegments[idx];

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				tempPS.color *= (materialColor * material.emittance);
				tempPS.remainingBounces = 0;
				//if (PATHCOMPACTION) {
				//	indixes[temp_idx] = 0;
				//}
			}
			// TODO: PBRT based shading function
			else {
				glm::vec3 intersectPoint = tempPS.ray.origin + intersection.t * tempPS.ray.direction;
				intersectPoint = intersectPoint + EPSILON * shadeableIntersections[idx].surfaceNormal;
				
				scatterRay(tempPS, intersectPoint, intersection.surfaceNormal, material, rng, intersection.outside);
				tempPS.remainingBounces--;
#if LASTBOUNCEDIRECTLIGHTING
				if (tempPS.remainingBounces == 0) {
					ShadeDirectLighting(rng, intersectPoint, shadeableIntersections[idx].surfaceNormal, geoms, geoms_size, light_indixes, no_of_lights, tempPS, materialColor, materials);
				}
#endif
			}
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
			//if (PATHCOMPACTION) {
			//	indixes[idx] = 0;
			//}
		}
	}
	else {
		//if (PATHCOMPACTION) {
		//	indixes[idx] = 0;
		//}
	}
}

__global__ void kernNaiveDirectLighting(int iter, int num_paths, ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials, int depth, int* indixes, int* light_indixes, int no_of_lights, Geom * geoms, int geoms_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pathSegments[idx].remainingBounces <= 0) return;
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			PathSegment& tempPS = pathSegments[idx];

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				tempPS.color *= (materialColor * material.emittance);
				tempPS.remainingBounces = 0;
			}
			else {
				glm::vec3 intersectPoint = tempPS.ray.origin + intersection.t * tempPS.ray.direction;
				intersectPoint = intersectPoint + EPSILON * shadeableIntersections[idx].surfaceNormal;

				ShadeDirectLighting(rng, intersectPoint, shadeableIntersections[idx].surfaceNormal, geoms, geoms_size, light_indixes, no_of_lights, tempPS, materialColor, materials);
				tempPS.remainingBounces = 0;
			}
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
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

/** Struct used to calculate if the path segment is valid or not by Thrust for Compaction
*	Verify implemenattion !!
*/
struct validaPath
{
	__host__ __device__
		bool operator()(const PathSegment &ps)
	{
		return ps.remainingBounces > 0;
	}
};

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

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, dev_indixes);
	checkCUDAError("generate camera ray");

	int depth = traceDepth;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

#if PERITERATIONTIMER
	auto startItr = std::chrono::high_resolution_clock::now();
#endif

	bool iterationComplete = false;
	while (!iterationComplete) {
#if PERDEPTHTIMER
		auto startDepth = std::chrono::high_resolution_clock::now();
#endif
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		if (CACHEFIRSTBOUNCE && iter > 1 && depth == traceDepth) {
			cudaMemcpy(dev_intersections, dev_intersectionsCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			// tracing
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_indixes
				, dev_MaterialSortPath
				, dev_MaterialSortIntersections
				, dev_materials
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();

			if (CACHEFIRSTBOUNCE && iter == 1 && depth == traceDepth) {
				cudaMemcpy(dev_intersectionsCache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}

#if SORTPATHSBYMATERIAL
		thrust::sort_by_key(thrust::device, dev_MaterialSortPath, dev_MaterialSortPath + num_paths, dev_paths);
		thrust::sort_by_key(thrust::device, dev_MaterialSortIntersections, dev_MaterialSortIntersections + num_paths, dev_intersections);
#endif

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory

#if NAIVEDIRECTLIGHTING
			kernNaiveDirectLighting << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				num_paths,
				dev_intersections,
				dev_paths,
				dev_materials,
				depth,
				dev_indixes,
				dev_light_indixes,
				no_of_lights,
				dev_geoms,
				hst_scene->geoms.size()
				);
			break;
#endif

		kernBRDFBasedShader<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth,
			dev_indixes,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_light_indixes,
			no_of_lights
		);

		if (depth == 0) {
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}

#if PATHCOMPACTION
			//num_paths = StreamCompaction::Efficient::compact(num_paths, dev_indixes, dev_indixes);
			PathSegment* partition_segments = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, validaPath());
			num_paths = partition_segments - dev_paths;
			iterationComplete = (num_paths == 0) || (depth == 0);
#endif

		depth--;
#if PERDEPTHTIMER
		auto endDepth = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> diffDepth = endDepth - startDepth;
		std::cout << "Elapsed Timer for Iteration " << iter << ", Depth " << depth << " : "<< diffDepth.count() << " ms \n";
#endif
	}

#if PERITERATIONTIMER
	auto endItr = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> diffItr = endItr - startItr;
	std::cout << "Elapsed Timer for Iteration " << iter << " : " << diffItr.count() << " ms \n";
#endif

	int original_num_path_size = dev_path_end - dev_paths;

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(original_num_path_size, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
