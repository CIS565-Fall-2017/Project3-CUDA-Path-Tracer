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
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "efficient.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "radix.h"
#include <fstream>

//Results file
extern FILE *fp;

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define Caching_Toggle 0
#define Sorting_Toggle 0
#define AntiAliasing_Toggle 1
#define MotionBlur_Toggle 1

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
static PathSegment * dev_cache_paths = NULL;
static ShadeableIntersection * dev_cache_intersections = NULL;
static int * dev_flag_array = NULL;
// ...

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
	//Cache
	cudaMalloc(&dev_cache_paths, pixelcount * sizeof(PathSegment));
	
	cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_flag_array, pixelcount * sizeof(int));
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_cache_paths);
	cudaFree(dev_cache_intersections);
	cudaFree(dev_flag_array);
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
		
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, x + y, 0);
#if AntiAliasing_Toggle
		float offset = 0.001;
		thrust::uniform_real_distribution<float> ux(-offset, offset);
		thrust::uniform_real_distribution<float> uy(-offset, offset);
		thrust::uniform_real_distribution<float> uz(-offset, offset);
		segment.ray.direction += glm::vec3(ux(rng), uy(rng), uz(rng));
#endif
		thrust::uniform_real_distribution<float> u01(0,1);
		segment.rand_time = u01(rng);
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__host__ __device__ glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
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

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];
#if MotionBlur_Toggle
			glm::vec3 blur_pos = glm::clamp((1 - pathSegment.rand_time) * geom.translation + pathSegment.rand_time * geom.translation_end, geom.translation, geom.translation_end);
			geom.transform = buildTransformationMatrix(blur_pos, geom.rotation, geom.scale);
			geom.inverseTransform = glm::inverse(geom.transform);
			geom.invTranspose = glm::inverseTranspose(geom.transform);
#endif
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

__global__ void shadeRealMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, glm::vec3 * image
	, int *flag_array
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment &this_Path = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			if (pathSegments[idx].remainingBounces) {
				flag_array[idx] = 1;
				scatterRay(this_Path, intersection.t * this_Path.ray.direction + this_Path.ray.origin, intersection.surfaceNormal, material, rng);
			}
			else {
				flag_array[idx] = 0;
				image[this_Path.pixelIndex] += this_Path.color;
			}
		}
		else {
			flag_array[idx] = 0;
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

__global__ void kernScatterPaths(int n, PathSegment *odata,
	const PathSegment *idata, const int *bools, const int *indices) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n)
		return;

	if (bools[index])
		odata[indices[index]] = idata[index];
}

int compact_Paths(int n) {
	// TODO
	if (n <= 0)
		return -1;
	int celllog = ilog2ceil(n);
	int pow2len = 1 << celllog;

	int *dev_indices;
	cudaMalloc((void**)&dev_indices, pow2len * sizeof(int));
	checkCUDAError("cudaMalloc dev_indices failed!");
	PathSegment *dev_temp_paths;
	cudaMalloc((void**)&dev_temp_paths, pow2len * sizeof(PathSegment));
	cudaMemcpy(dev_temp_paths, dev_paths, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

	// Scan
	cudaMemcpy(dev_indices, dev_flag_array, n * sizeof(int), cudaMemcpyDeviceToDevice);
	checkCUDAError("cudaMemcpy failed!");

	int blockSize = 128;
	int blockNum;

	//Up-Sweep
	for (int d = 0; d <= celllog - 1; d++) {
		int interval_length = (1 << (d + 1));
		blockNum = (pow2len / interval_length + blockSize) / blockSize;
		StreamCompaction::Efficient::cudaSweepUp << <blockNum, blockSize >> >(pow2len / interval_length, d, dev_indices);
	}
	//Down-Sweep
	cudaMemset(dev_indices + pow2len - 1, 0, sizeof(int));
	checkCUDAError("cudaMemset failed!");

	for (int d = celllog - 1; d >= 0; d--) {
		int interval_length = (1 << (d + 1));
		blockNum = (pow2len / interval_length + blockSize) / blockSize;
		StreamCompaction::Efficient::cudaSweepDown << <blockNum, blockSize >> >(pow2len / interval_length, d, dev_indices);
	}


	//Scattered
	blockNum = (n + blockSize) / blockSize;
	kernScatterPaths << <blockNum, blockSize >> >(n, dev_paths, dev_temp_paths, dev_flag_array, dev_indices);

	//compute count
	int a, b;
	cudaMemcpy(&a, dev_flag_array + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&b, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
	int count = a + b;

	//Free data
	cudaFree(dev_indices);
	checkCUDAError("cudaFree dev_idata failed!");
	cudaFree(dev_temp_paths);
	checkCUDAError("cudaFree dev_temp_paths failed!");

	return count;
}

void compressedPathandIntersection(int& num_paths, PathSegment *paths, int *flag)
{
	thrust::device_ptr<int> dev_ptrFlag(flag);
	thrust::device_ptr<PathSegment> dev_ptrPaths(paths);
	thrust::remove_if(dev_ptrPaths, dev_ptrPaths + num_paths, dev_ptrFlag, thrust::logical_not<int>());
	num_paths = thrust::count_if(dev_ptrFlag, dev_ptrFlag + num_paths, thrust::identity<int>());
}

// Sort by materialId
typedef thrust::tuple<PathSegment, ShadeableIntersection> Tuple;
class cmp
{
public:
	__host__ __device__ bool operator()(const Tuple &a, const Tuple &b)
	{
		return a.get<1>().materialId < b.get<1>().materialId;
	}
};

void sortByMaterialId(int num_paths, PathSegment *dev_paths, ShadeableIntersection *dev_intersections)
{
	thrust::device_ptr<PathSegment> ptrPath(dev_paths);
	thrust::device_ptr<ShadeableIntersection> ptrIntersection(dev_intersections);

	typedef thrust::tuple<thrust::device_ptr<PathSegment>, thrust::device_ptr<ShadeableIntersection>> IteratorTuple;
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	ZipIterator zip_begin = thrust::make_zip_iterator(thrust::make_tuple(ptrPath, ptrIntersection));
	ZipIterator zip_end = zip_begin + num_paths;
	thrust::sort(zip_begin, zip_end, cmp());
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
// -------------- Caching --------------
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	StreamCompaction::Efficient::timer().startGpuTimer();
	

#if Caching_Toggle && !AntiAliasing_Toggle
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(dev_cache_paths, dev_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		checkCUDAError("Memcpy dev_paths to dev_cache_paths");
		// clean shading chunks
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
		cudaMemcpy(dev_cache_intersections, dev_intersections, num_paths * sizeof(ShadeableIntersection),  cudaMemcpyDeviceToDevice);
		checkCUDAError("Memcpy dev_intersections to dev_cache_intersections");
	}	
#endif	 
	

// --------------- PathSegment Tracing Stage -----------------
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	bool firstStage = true;

	while (!iterationComplete) {
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		
		if (firstStage) {
			firstStage = false;
#if Caching_Toggle && !AntiAliasing_Toggle
			cudaMemcpy(dev_paths, dev_cache_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_intersections, dev_cache_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
#else
			generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
			checkCUDAError("generate camera ray");
			// clean shading chunks
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
#endif
		}
		else {
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
		}
		depth++;


	// TODO:
// ---------------- Shading Stage ------------------
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.
		

		//The last 
		//int num_materials = 5;
		//StreamCompaction::Radix::RadixSort_Path_Interactions(num_paths, dev_paths, dev_intersections, num_materials);
#if Sorting_Toggle
		sortByMaterialId(num_paths, dev_paths, dev_intersections);
#endif
		shadeRealMaterial << <numblocksPathSegmentTracing, blockSize1d >> >(
			iter, 
			num_paths, 
			dev_intersections, 
			dev_paths, 
			dev_materials, 
			dev_image, 
			dev_flag_array);
		//get new paths and new flag_array
// ----------------- Stream Compaction ----------------
		num_paths = compact_Paths(num_paths);
		//compressedPathandIntersection(num_paths, dev_paths, dev_flag_array);
		//get new path pool and num_paths
		if(num_paths <= 0)
			iterationComplete = true; // TODO: should be based off stream compaction results.
	}

	// Assemble this iteration and apply it to the image
	/*dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);*/
///////////////////////////////////////////////////////////////////////////

	StreamCompaction::Efficient::timer().endGpuTimer();
	//std::cout << "	elapsed time: " << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << " ms\n";
	//fprintf(fp, "%lf\n", StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
