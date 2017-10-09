#pragma once
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
#include "glm/gtx/component_wise.hpp"
#include "utilities.h"
#include "utilkern.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "integrators.h"

#include "stream_compaction\sharedandbank.h" 
#include "stream_compaction\radix.h" 

#define ERRORCHECK			1 
#define COMPACT				1 //0 NONE, 1 THRUST, 2 CUSTOM(breaks render, just use for timing compare)
#define PT_TECHNIQUE		1 //0 NAIVE, 1 MIS, 2 Multikern MIS(slower)
#define FIRSTBOUNCECACHING	0
#define TIMER				0
#define MATERIALSORTING		1 
//https://thrust.1ithub.io/doc/group__stream__compaction.html#ga5fa8f86717696de88ab484410b43829b
//https://stackoverflow.com/questions/34103410/glmvec3-and-epsilon-comparison
struct isDead { //needed for thrust's predicate, the last arg in remove_if
	__host__ __device__
		bool operator()(const PathSegment& path) { //keep the true cases
		return (path.remainingBounces > 0);
	}
};

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
	std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}
//ALREADY DEFINED IN STREAM_COMPACTION
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
static int* dev_geomLightIndices = NULL;
static ShadeableIntersection * dev_firstbounce = NULL;
static int* dev_materialIDsForPathsSort = NULL;
static int* dev_materialIDsForIntersectionsSort = NULL;

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

  	cudaMalloc(&dev_firstbounce, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_firstbounce, 0, pixelcount * sizeof(ShadeableIntersection));
    // TODO: initialize any extra device memeory you need
	//check which geoms are emissive and copy them to its own array(for large scenes 
	//rather not pass geoms if we don't have to)
	std::vector<int> geomLightIndices;
	for (int i = 0; i < scene->geoms.size(); ++i) {
		int materialID = scene->geoms[i].materialid;
		if (scene->materials[materialID].emittance > 0) {
			geomLightIndices.push_back(i);
			numlights++;
		}
	}
	cudaMalloc((void**)&dev_geomLightIndices, numlights * sizeof(int));
  	cudaMemcpy(dev_geomLightIndices, geomLightIndices.data(), numlights * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_materialIDsForPathsSort, pixelcount * sizeof(int));
	cudaMalloc((void**)&dev_materialIDsForIntersectionsSort, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_geomLightIndices);
	numlights = 0;
  	cudaFree(dev_firstbounce);
	cudaFree(dev_materialIDsForPathsSort);
	cudaFree(dev_materialIDsForIntersectionsSort);

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

	// TODO: implement antialiasing by jittering the ray
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u(-0.5f, 0.5f);

	//camera top right is (0,0)
#if 1 == FIRSTBOUNCECACHING
	segment.ray.direction = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * (x - cam.resolution.x * 0.5f)
		- cam.up    * cam.pixelLength.y * (y - cam.resolution.y * 0.5f));
#else
	segment.ray.direction = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * (x - cam.resolution.x * 0.5f + u(rng))
		- cam.up    * cam.pixelLength.y * (y - cam.resolution.y * 0.5f + u(rng)));
#endif

	segment.MSPaintPixel = glm::ivec2(cam.resolution.x - 1 - x, y);
	segment.pixelIndex = index;
	segment.remainingBounces = traceDepth;
	segment.color = glm::vec3(1.f);

#if 1 == PT_TECHNIQUE || 2 == PT_TECHNIQUE //MIS
	segment.color = glm::vec3(0.f);
	segment.throughput = glm::vec3(1.f);
	segment.specularbounce = false;
#endif
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(const int iter, const int depth,
	const int num_paths, PathSegment * pathSegments,
	const Geom* const geoms, const int geoms_size,
	ShadeableIntersection* const intersections,
	ShadeableIntersection* const firstbounce, const int firstbouncecaching) 
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int actual_first_iter = 1;//iter 0 is an initialization step

	if (path_index >= num_paths) { return; }
	PathSegment pathSegment = pathSegments[path_index];

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;//why is this needed if the normal is being corrected already

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	//FIRSTBOUNCECACHING
	// naive parse through global geoms
	if ((1 == firstbouncecaching && 0 == depth && actual_first_iter == iter) || (0 < depth) || (0 == firstbouncecaching)) {
		for (int i = 0; i < geoms_size; i++) {
			const Geom & geom = geoms[i];
			t = shapeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

			//min
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}//for i < geoms_size
	}//if we need to find closest intersection

	if (1 == firstbouncecaching && 0 == depth && actual_first_iter == iter) {//first time we call computeintersection
		if (-1 == hit_geom_index) {
			firstbounce[path_index].t = -1.f;
			intersections[path_index].t = -1.f;
		} else {
			firstbounce[path_index].t = t_min;
			firstbounce[path_index].materialId = geoms[hit_geom_index].materialid;
			firstbounce[path_index].surfaceNormal = normal;
			firstbounce[path_index].geomId = hit_geom_index;

			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].geomId = hit_geom_index;
		}
	} else if (0 == depth && actual_first_iter < iter && 1 == firstbouncecaching) { //first depth in other iters when firstbounce is enabled
		intersections[path_index].t = firstbounce[path_index].t;
		intersections[path_index].materialId = firstbounce[path_index].materialId;
		intersections[path_index].surfaceNormal = firstbounce[path_index].surfaceNormal;
		intersections[path_index].geomId = firstbounce[path_index].geomId;
	} else {//first bounce is off or depth is greater than 0, do it normally
		if (-1 == hit_geom_index) {
			intersections[path_index].t = -1.f;
		} else {
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].geomId = hit_geom_index;
		}
	}
}
// Add the current iteration's output to the overall image
__global__ void finalGather(const int nPaths, const glm::ivec2 resolution, 
	glm::vec3 * image, const PathSegment * iterationPaths)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nPaths) { return; }

	PathSegment iterationPath = iterationPaths[index];
	image[iterationPath.pixelIndex] += iterationPath.color;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) { //, const int MAXBOUNCES) {
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
	

#if 1 == TIMER
	using time_point_t = std::chrono::high_resolution_clock::time_point;
	time_point_t time_start_cpu = std::chrono::high_resolution_clock::now();
#endif
	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks, prob only needs to be num_paths not pixelcount
		//cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		//checkCUDAError("cuda memset dev_intersections");

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), 
			dev_intersections, dev_firstbounce, FIRSTBOUNCECACHING
			);
		checkCUDAError("trace one bounce");
		//cudaDeviceSynchronize();
#if 1 == MATERIALSORTING
	//copy material ids to to two dev material id arrays. 1 needed for sorting paths and 1 for isects.
		//our threadid for shading is used to index into dev_intersections and dev_paths
		copyMaterialIDsToArrays << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths, dev_materialIDsForIntersectionsSort, dev_materialIDsForPathsSort, dev_intersections);
	//thrust sort sorts the second array by how it sorted the first array.
		thrust::sort_by_key(thrust::device, dev_materialIDsForIntersectionsSort, dev_materialIDsForIntersectionsSort + num_paths, dev_intersections);
		thrust::sort_by_key(thrust::device, dev_materialIDsForPathsSort, dev_materialIDsForPathsSort + num_paths, dev_paths);
#endif


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if 0 == PT_TECHNIQUE
		shadeMaterialNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, num_paths, dev_intersections, dev_paths, dev_materials, dev_geoms
			);
		checkCUDAError("shadeMaterial");
#elif 1 == PT_TECHNIQUE
		shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth, numlights, traceDepth, num_paths, dev_intersections, dev_paths,
			dev_materials, dev_geomLightIndices, dev_geoms, hst_scene->geoms.size()
			);
		checkCUDAError("shadeMaterial");
#elif 2 == PT_TECHNIQUE
		shadeMaterialMIS_DLlight << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth, numlights, traceDepth, num_paths, dev_intersections, dev_paths,
			dev_materials, dev_geomLightIndices, dev_geoms, hst_scene->geoms.size()
			);
		checkCUDAError("shadeMaterial");

		shadeMaterialMIS_DLbxdf << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth, numlights, traceDepth, num_paths, dev_intersections, dev_paths,
			dev_materials, dev_geomLightIndices, dev_geoms, hst_scene->geoms.size()
			);
		checkCUDAError("shadeMaterial");

		shadeMaterialMIS_throughput << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth, numlights, traceDepth, num_paths, dev_intersections, dev_paths,
			dev_materials, dev_geomLightIndices, dev_geoms, hst_scene->geoms.size()
			);
				//gpuErrchk(cudaPeekAtLastError());
				//gpuErrchk(cudaDeviceSynchronize());
		checkCUDAError("shadeMaterial");
#else
		printf("\n unknown PT_TECHNIQUE \n");
#endif

		depth++;

		//Do compact and determine how many paths are still left, our compact only operates on ints...so use thrust
#if 0 == COMPACT
		if (traceDepth == depth) { iterationComplete = true; num_paths = 0; }
#elif 1 == COMPACT
		//PathSegment *compactend = thrust::remove_if(dev_paths, dev_paths + num_paths, isDead());
		PathSegment *compactend = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isDead());
		checkCUDAError("thrust::partition");
		num_paths = compactend - dev_paths;
		if (0 == num_paths) { iterationComplete = true; }
#elif 2 == COMPACT
		num_paths = StreamCompaction::SharedAndBank::compactNoMalloc_PathSegment(num_paths, dev_paths);
		checkCUDAError("stream compaction");
		//printf("\nnum_paths: %i", num_paths);
		if (0 == num_paths) { iterationComplete = true; }
#else
		printf("\n UKNOWN COMPACT setting \n");
#endif

	}//////////////////////END WHILE


#if 1 == TIMER
	cudaDeviceSynchronize();
	time_point_t time_end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
	float prev_elapsed_time_cpu_milliseconds = static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());
	printElapsedTime(prev_elapsed_time_cpu_milliseconds, "(std::chrono Measured)");
#endif

	// Assemble this iteration and apply it to the image
	num_paths = dev_path_end - dev_paths;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, cam.resolution, dev_image, dev_paths);
    checkCUDAError("final gather");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    checkCUDAError("send image to pixel buffer object GL");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("get dev_image to cpu for saving");
}
