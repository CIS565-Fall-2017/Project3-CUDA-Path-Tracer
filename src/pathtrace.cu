#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <stdlib.h> 
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include <chrono>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "sampling.h"

#define ERRORCHECK 1

//--------------------
//Toggle-able OPTIONS
//--------------------
#define FIRST_BOUNCE_INTERSECTION_CACHING 0
#define MATERIAL_SORTING 0
#define INACTIVE_RAY_CULLING 1 //Stream Compaction of rays
#define DEPTH_OF_FIELD 0
#define ANTI_ALIASING 1 // if you use first bounce caching, antialiasing will not work --> can make it work with 
						// a more deterministic Supersampling approach to AA but requires more memory for 
						// caching more intersections
//Naive Integration is the default if nothing else is toggled
#define DIRECT_LIGHTING_INTEGRATOR 0
#define FULL_LIGHTING_INTEGRATOR 0

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

//------------------------------------------------
//-------------------Timer------------------------
using time_point_t = std::chrono::high_resolution_clock::time_point;
time_point_t timeStartCpu;
time_point_t timeEndCpu;
float prevElapsedTime = 0.0f;

void printTimerDetails(int iteration, int depth, float time)
{
	printf("%f\n", time);
	//printf("Iteration %d; Depth %d: %f milliseconds\n", iteration, depth, time);
}
//------------------------------------------------

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) 
{
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

// Static Variables
static Scene * hst_scene = NULL;
static Color3f * dev_image = NULL;
//Geometries, Lights, and Materials
static Geom * dev_geoms = NULL; //Geoms array contains all the lights in the scene as well
static Light * dev_lights = NULL;
static Material * dev_materials = NULL;
//PathSegments and Intersections
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
//Iterative Integration Necessary Variables
static int * dev_rayPixelIndex = NULL; // dont want to sort more and more arrays --> so just sort this one
									   // also helpful when moving to a Struct of Arrays format	
static Color3f * dev_accumulatedThroughputColor = NULL;
static Color3f * dev_accumulatedRayColor = NULL;
static Color3f * dev_totalPixelColor = NULL;
// Cache First Bounce
static int *dev_intersectionsCached = NULL;
// Sort by material
static int *dev_sortingPixelIndices = NULL;
static int *dev_pathMaterialIndices = NULL;
static int *dev_intersectionMaterialIndices = NULL;

__global__ void scaleDownLightIntensity(int nLights, Light * lights, Geom * geoms, Material * materials)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nLights)
	{
		int mat_id = geoms[lights[index].lightGeomIndex].materialid;
		materials[mat_id].emittance /= 40.0f;
	}
}

void pathtraceInit(Scene *scene) 
{
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	
	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	//To access the correct pixel after sorting
	cudaMalloc(&dev_rayPixelIndex, pixelcount * sizeof(int));
	//Iterative Integration Necessary Variables
	cudaMalloc(&dev_totalPixelColor, pixelcount * sizeof(Color3f));
	cudaMemset(dev_totalPixelColor, 0, pixelcount * sizeof(Color3f));
	cudaMalloc(&dev_accumulatedRayColor, pixelcount * sizeof(Color3f));
	cudaMemset(dev_accumulatedRayColor, 0, pixelcount * sizeof(Color3f));
	cudaMalloc(&dev_accumulatedThroughputColor, pixelcount * sizeof(Color3f));
	cudaMemset(dev_accumulatedThroughputColor, 1, pixelcount * sizeof(Color3f));

	// Caching the first Intersection for evey iteration
	cudaMalloc(&dev_intersectionsCached, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersectionsCached, 0, pixelcount * sizeof(ShadeableIntersection));
	
	// sort Paths by material
	cudaMalloc(&dev_pathMaterialIndices, pixelcount * sizeof(int));
	// sort Intersections by material
	cudaMalloc(&dev_intersectionMaterialIndices, pixelcount * sizeof(int));
	// sorting pixelIndices array in accordance to how pathsegments were sorted
	cudaMalloc(&dev_sortingPixelIndices, pixelcount * sizeof(int));

#if FULL_LIGHTING_INTEGRATOR
	int numLights = hst_scene->lights.size();
	const int blockSize1d = 128;
	dim3 numblocksPathSegmentTracing = (numLights + blockSize1d - 1) / blockSize1d;
	scaleDownLightIntensity <<<numLights, blockSize1d>>> (numLights, dev_lights, dev_geoms, dev_materials);
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() 
{
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
	cudaFree(dev_lights);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	
	cudaFree(dev_rayPixelIndex);

	cudaFree(dev_accumulatedRayColor);
	cudaFree(dev_accumulatedThroughputColor);
	cudaFree(dev_totalPixelColor);

	cudaFree(dev_intersectionsCached);
	cudaFree(dev_sortingPixelIndices);
	cudaFree(dev_pathMaterialIndices);
	cudaFree(dev_intersectionMaterialIndices);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int * pixelIndices)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) 
	{
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = Color3f(1.0f, 1.0f, 1.0f);
		segment.accumulatedThroughput = Color3f(1.0f, 1.0f, 1.0f);
		segment.accumulatedColor = Color3f(0.0f, 0.0f, 0.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
#if ANTI_ALIASING
		// Antialiasing by jittering the ray direction using the offsets
		thrust::uniform_real_distribution<float> u01(-1, 1);
		float x_offset = u01(rng);
		float y_offset = u01(rng);

		segment.ray.direction = glm::normalize(cam.view
											   - cam.right * cam.pixelLength.x * ((float)x - x_offset - (float)cam.resolution.x * 0.5f)
											   - cam.up * cam.pixelLength.y * ((float)y - y_offset - (float)cam.resolution.y * 0.5f)
											  );
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

#if DEPTH_OF_FIELD
		// Dynamically adjustable Depth of Field
		if (cam.lensRadius>EPSILON)
		{
			//picking a samplepoint on the lens and then changing it according to the lens properties
			thrust::uniform_real_distribution<float> u02(0, 1);
			Point2f xi = Point2f(u02(rng), u02(rng));
			Point3f pLens = cam.lensRadius * sampling_SquareToDiskConcentric(xi);
			Point3f pFocus = cam.focalDistance * segment.ray.direction + segment.ray.origin;
			Point3f aperaturePoint = segment.ray.origin + (cam.up * pLens.y) + (cam.right * pLens.x);

			segment.ray.origin = aperaturePoint;
			segment.ray.direction = glm::normalize(pFocus - aperaturePoint);
		}
#endif

		segment.pixelIndex = index;
		pixelIndices[index] = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
__global__ void computeIntersections(int numActiveRays, PathSegment * pathSegments, Geom * geoms, 
									int geoms_size, ShadeableIntersection * intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < numActiveRays)
	{
		computeIntersectionsForASingleRay(pathSegments[path_index].ray, intersections[path_index], geoms, geoms_size);
	}
}

__global__ void sortIndicesByMaterial(int numActiveRays, ShadeableIntersection *intersections, 
									  int *pathIndices, int *intersectionIndices, int *rayIndices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numActiveRays) {
		int materialId = intersections[idx].materialId;
		pathIndices[idx] = materialId;
		intersectionIndices[idx] = materialId;
		rayIndices[idx] = materialId;
	}
}

__global__ void shadeMaterialsNaive(int iter, int numActiveRays, ShadeableIntersection * shadeableIntersections,
									PathSegment * pathSegments, Geom * geoms, int numGeoms, Material * materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numActiveRays)
	{
		if (pathSegments[idx].remainingBounces <= 0)
		{
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		Material material = materials[intersection.materialId];

		//If the ray didnt intersect with objects in the scene
		if (intersection.t < 0.0f)
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0; //to make thread stop executing things
			return;
		}

		//If the ray hit a light in the scene
		if (material.emittance > 0.0f)
		{
			pathSegments[idx].color *= material.color*material.emittance;
			pathSegments[idx].remainingBounces = 0; //equivalent of breaking out of the thread
			return;
		}

		// Set up the RNG
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

		// if the intersection exists and the itersection is not a light then
		//deal with the material and end up changing the pathSegment color and its ray direction
		naiveIntegrator(pathSegments[idx], intersection, material, 
						geoms[shadeableIntersections[idx].hitGeomIndex], 
						geoms, numGeoms, rng);
	}
}

__global__ void shadeMaterialsDirect(int iter, int numActiveRays, ShadeableIntersection * shadeableIntersections,
									PathSegment * pathSegments, Material * materials, Geom * geoms, int numGeoms,
									Light * lights, int numLights, Color3f * accumulatedColor)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numActiveRays)
	{
		if (pathSegments[idx].remainingBounces <= 0)
		{
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		Material material = materials[intersection.materialId];

		//If the ray didnt intersect with objects in the scene
		if (intersection.t < 0.0f)
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0; //to make thread stop executing things
			return;
		}

		//If the ray hit a light in the scene
		if (material.emittance > 0.0f)
		{
			pathSegments[idx].color *= material.color*material.emittance;
			pathSegments[idx].remainingBounces = 0; //equivalent of breaking out of the thread
			return;
		}

		// Set up the RNG
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

		// if the intersection exists and the itersection is not a light then
		//deal with the material and end up changing the pathSegment color and its ray direction
		directLightingIntegrator(pathSegments[idx], intersection, materials, material,
								geoms[shadeableIntersections[idx].hitGeomIndex], geoms, numGeoms, 
								lights, numLights, rng);
	}
}

__global__ void shadeMaterialsFullLighting(int iter, int maxtraceDepth, int numActiveRays,
										   ShadeableIntersection * shadeableIntersections,
										   PathSegment * pathSegments, 
										   Material * materials, 
										   Geom * geoms, int numGeoms,
										   Light * lights, int numLights, 
										   Color3f * accumulatedThroughput, Color3f * accumulatedColor )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numActiveRays)
	{
		if (pathSegments[idx].remainingBounces <= 0)
		{
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		Material material = materials[intersection.materialId];
		Color3f rayAccumulatedThroughput = accumulatedThroughput[idx];
		Color3f rayAccumulatedColor = accumulatedColor[idx];

		//If the ray didnt intersect with objects in the scene
		if (intersection.t < 0.0f)
		{
			rayAccumulatedColor = Color3f(0.0f);
			pathSegments[idx].remainingBounces = 0; //to make thread stop executing things
			return;
		}

		//If the ray hit a light in the scene
		if (material.emittance > 0.0f)
		{
			pathSegments[idx].accumulatedColor += material.color*material.emittance;
			pathSegments[idx].remainingBounces = 0; //equivalent of breaking out of the thread
			return;
		}

		// Set up the RNG
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

		// if the intersection exists and the itersection is not a light then
		//deal with the material and end up changing the pathSegment color and its ray direction
		fullLightingIntegrator(maxtraceDepth, 
							pathSegments[idx], intersection, 
							materials, material,
							geoms, geoms[shadeableIntersections[idx].hitGeomIndex], numGeoms, 
							lights, numLights,
							rayAccumulatedThroughput, rayAccumulatedColor,
							rng);
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
#if FULL_LIGHTING_INTEGRATOR
		image[iterationPath.pixelIndex] += iterationPath.accumulatedColor;
#else
		image[iterationPath.pixelIndex] += iterationPath.color;
#endif
	}
}

__global__ void averagePixelColor(int nPaths, Color3f * image, PathSegment * iterationPaths,
									int * pixelIndices, int iter,
									Color3f * accumulatedColor, Color3f * totalPixelColor)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		//int pixelIndex = pixelIndices[index];
		int pixelIndex = iterationPaths[index].pixelIndex;
		totalPixelColor[pixelIndex] += iterationPath.accumulatedColor;
		image[pixelIndex] = totalPixelColor[pixelIndex];// float(iter);
	}
}

struct predicate_RemainingBounces
{
	__host__ __device__ bool operator()(const PathSegment &x)
	{
		return (x.remainingBounces > 0);
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) 
{
	// This function performs one iteration of the path tracing algorithm
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d( (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
								(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y );
	// 1D block for path tracing
	const int blockSize1d = 128;

	//print lights details
	//printf("light: %d \n", hst_scene->lights[0].lightGeomIndex);
	//printf("number of lights: %d \n", hst_scene->lights.size());

	//------------------------------------------------
	//Timer Start
	//timeStartCpu = std::chrono::high_resolution_clock::now();
	//------------------------------------------------

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * Stream compact away all of the terminated paths.
	//   * Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	//	 * Finally, add this iteration's results to the image.

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, dev_rayPixelIndex);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	//------------------------------------------------
	// --- PathSegment Tracing Stage ---
	//------------------------------------------------
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	int activeRays = num_paths;
	
#if FULL_LIGHTING_INTEGRATOR
	// reset accumulatedThroughputColor and accumulatedRayColor for fullLighting
	cudaMemset(dev_accumulatedRayColor, 0.0f, pixelcount * sizeof(Color3f));
	cudaMemset(dev_accumulatedThroughputColor, 1.0f, pixelcount * sizeof(Color3f));
#elif DIRECT_LIGHTING_INTEGRATOR
	//cudaMemset(dev_accumulatedRayColor, 0.0f, pixelcount * sizeof(Color3f));
#endif

	while (!iterationComplete) 
	{
		dim3 numblocksPathSegmentTracing = (activeRays + blockSize1d - 1) / blockSize1d;

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		//------------------------------------------------
		//Compute Intersections with an option for Intersection Caching for the first bounce
		//------------------------------------------------
#if FIRST_BOUNCE_INTERSECTION_CACHING
		//Checking if intersection cached results should be used
		if(iter == 1 && depth == 0)
		{
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (activeRays, dev_paths, dev_geoms,
																				 hst_scene->geoms.size(), dev_intersections);
			checkCUDAError("compute Intersections Failed");
			cudaDeviceSynchronize();

			//Copy cached intersections
			cudaMemcpy(dev_intersectionsCached, dev_intersections, activeRays * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (iter != 1 && depth == 0)
		{
			//Cache the first Bounce
			cudaMemcpy(dev_intersections, dev_intersectionsCached, activeRays * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}

		if (depth > 0)
		{
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (activeRays, dev_paths, dev_geoms,
																				hst_scene->geoms.size(), dev_intersections);
			checkCUDAError("compute Intersections Failed");
			cudaDeviceSynchronize();
		}
#else
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (activeRays, dev_paths, dev_geoms,
																			 hst_scene->geoms.size(), dev_intersections);
		checkCUDAError("compute Intersections Failed");
		cudaDeviceSynchronize();
#endif
		
		depth++;

#if MATERIAL_SORTING
		//------------------------------------------------
		//Sort Rays By Material
		//------------------------------------------------
		sortIndicesByMaterial <<<numblocksPathSegmentTracing, blockSize1d>>> (activeRays, dev_intersections, 
																			  dev_pathMaterialIndices, 
																			  dev_intersectionMaterialIndices,
																			  dev_sortingPixelIndices);
		
		thrust::sort_by_key(thrust::device, dev_pathMaterialIndices, dev_pathMaterialIndices + activeRays, dev_paths);
		thrust::sort_by_key(thrust::device, dev_intersectionMaterialIndices, dev_intersectionMaterialIndices + activeRays, dev_intersections);
		thrust::sort_by_key(thrust::device, dev_sortingPixelIndices, dev_sortingPixelIndices + activeRays, dev_rayPixelIndex);
#endif

		//------------------------------------------------
		//Integration Schemes --> They Color Path Segments and trace Rays
		//------------------------------------------------
#if DIRECT_LIGHTING_INTEGRATOR
		shadeMaterialsDirect <<<numblocksPathSegmentTracing, blockSize1d>>> (iter, activeRays, dev_intersections,
																			 dev_paths, dev_materials, 
																			 dev_geoms, hst_scene->geoms.size(), 
																			 dev_lights, hst_scene->lights.size(),
																			 dev_accumulatedRayColor);
#elif FULL_LIGHTING_INTEGRATOR
		shadeMaterialsFullLighting <<<numblocksPathSegmentTracing, blockSize1d>>> (iter, traceDepth, activeRays, 
																				dev_intersections, dev_paths, dev_materials,
																				dev_geoms, hst_scene->geoms.size(),
																				dev_lights, hst_scene->lights.size(),
																				dev_accumulatedThroughputColor, 
																				dev_accumulatedRayColor);
#else // NAIVE_INTEGRATOR
		shadeMaterialsNaive <<<numblocksPathSegmentTracing, blockSize1d>>> (iter, activeRays, dev_intersections, 
																			dev_paths, dev_geoms, hst_scene->geoms.size(),
																			dev_materials);
		checkCUDAError("Naive Integration Scheme Failed");
		cudaDeviceSynchronize();
#endif

#if INACTIVE_RAY_CULLING		
		//------------------------------------------------
		// Stream Compact your array of rays to cull out rays that are no longer active
		//------------------------------------------------
		//thrust::partition returns a pointer to the element in the array where the partition occurs 

		PathSegment* partition_point = thrust::partition(thrust::device, dev_paths, dev_paths + activeRays, predicate_RemainingBounces());
		activeRays = partition_point - dev_paths;
#endif

		if (depth >= traceDepth || activeRays <= 0)// based off stream compaction results.
		{
			iterationComplete = true;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather <<<numBlocksPixels, blockSize1d>>> (num_paths, dev_image, dev_paths);
	//------------------------------------------------
	//Timer End
	//timeEndCpu = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double, std::milli> duration = timeEndCpu - timeStartCpu;
	//prevElapsedTime = static_cast<decltype(prevElapsedTime)>(duration.count());
	//printTimerDetails(iter, depth, prevElapsedTime);
	//------------------------------------------------

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
}
