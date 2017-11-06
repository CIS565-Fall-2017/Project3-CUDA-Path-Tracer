#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "stream_compaction/efficient_shared.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

// optimization flags
#define COMPACT 1
#define CONTIG_MAT 0
#define CACHE_FIRST 0
#define BV_CULLING 1

// feature flags
#define ANTI_ALIAS 1
#define DEPTH_OF_FIELD 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
//void checkCUDAErrorFn(const char *msg, const char *file, int line) {
//#if ERRORCHECK
//	cudaDeviceSynchronize();
//	cudaError_t err = cudaGetLastError();
//	if (cudaSuccess == err) {
//		return;
//	}
//
//	fprintf(stderr, "CUDA error");
//	if (file) {
//		fprintf(stderr, " (%s:%d)", file, line);
//	}
//	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
//#  ifdef _WIN32
//	getchar();
//#  endif
//	exit(EXIT_FAILURE);
//#endif
//}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__host__ __device__
glm::vec3 squareToDiskConcentric(float x, float y) 
{
	float r, phi;
	float PiOver4 = 0.78539816339744830961;
	float a = 2.f * x - 1.f; // map to [-1, 1]
	float b = 2.f * y - 1.f;

	if (a > -b) {
		if (a > b) {
			r = a;
			phi = PiOver4 * b / a;
		}
		else {
			r = b;
			phi = PiOver4 * (2.f - a / b);
		}
	}
	else {
		if (a < b) {
			r = -a;
			phi = PiOver4 * (4.f + b / a);
		}
		else {
			r = -b;
			if (b != 0.f) phi = PiOver4 * (6.f - a / b);
			else phi = 0.f;
		}
	}
	return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), 0.f);
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
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

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
static int * dev_indices = NULL;
static int * dev_sort_mat = NULL;
static ShadeableIntersection * dev_first_intersections = NULL;

bool cpu_timer_started = false;
float prev_elapsed_time_cpu_milliseconds = 0.f;
std::chrono::high_resolution_clock::time_point time_start_cpu;
std::chrono::high_resolution_clock::time_point time_end_cpu;

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
	cudaMalloc(&dev_indices, pixelcount * sizeof(int));
	cudaMalloc(&dev_sort_mat, pixelcount * sizeof(int));
	cudaMemset(dev_sort_mat, scene->materials.size(), pixelcount * sizeof(int));

	cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

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
	cudaFree(dev_sort_mat);
	cudaFree(dev_first_intersections);

	checkCUDAError("pathtraceFree");
}

void startCpuTimer()
{
	if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
	cpu_timer_started = true;

	time_start_cpu = std::chrono::high_resolution_clock::now();
}

void endCpuTimer()
{
	time_end_cpu = std::chrono::high_resolution_clock::now();

	if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

	std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
	prev_elapsed_time_cpu_milliseconds =
		static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

	cpu_timer_started = false;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int* indices)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {

		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		float dx = 0;
		float dy = 0;

#if ANTI_ALIAS
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
		thrust::uniform_real_distribution<float> u01(0, 1);

		dx = u01(rng);
		dy = u01(rng);
#endif

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + dx - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + dy - (float)cam.resolution.y * 0.5f)
		);

#if DEPTH_OF_FIELD
		float focalD = 11;
		float lensR = 0.8;
		float t = glm::abs(focalD / segment.ray.direction.z);
		glm::vec3 pFocus = t * segment.ray.direction;
		glm::vec3 origin = lensR * squareToDiskConcentric(u01(rng), u01(rng));
		segment.ray.direction = glm::normalize(pFocus - origin);
		segment.ray.origin += origin;

#endif
		segment.pixelIndex = index;
		indices[index] = index + 1;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment * pathSegments,
	int * indices,
	int * mats,
	Geom * geoms,
	int geoms_size,
	ShadeableIntersection * intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		int index = indices[path_index] - 1;
		PathSegment pathSegment = pathSegments[index];

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
			else if (geom.type == TRIANGLE)
			{
				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == MESH)
			{
#if BV_CULLING
				bool hit = bbIntersectionTest(geom, pathSegment.ray);
				if (!hit) {
					i += geom.nextIdxOff;
				}
#endif
				continue;
			}
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
			intersections[index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[index].t = t_min;
			int mat = geoms[hit_geom_index].materialid;
			intersections[index].materialId = mat;
			mats[path_index] = mat;
			intersections[index].surfaceNormal = normal;
			intersections[index].surfacePoint = intersect_point;
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
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials)
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
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeMaterial(
	int iter,
	int num_paths,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	int * indices,
	Material * materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int index = indices[idx] - 1;

	bool notTerminated = false;
#if COMPACT
	notTerminated = indices[idx] > 0;
#else
	notTerminated = pathSegments[index].remainingBounces > 0;
#endif
		
	if (notTerminated) {
		ShadeableIntersection intersection = shadeableIntersections[index];
		PathSegment *pathSeg = &pathSegments[index];
		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSeg->remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSeg->color *= (materialColor * material.emittance);
				pathSeg->remainingBounces = 0;
			}
			// TODO : lighting computation
		
			else {
				scatterRay(*pathSeg, intersection.surfacePoint, intersection.surfaceNormal, material, rng);
			}
		}
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
		else {
			pathSeg->color = glm::vec3(0.0f);
			pathSeg->remainingBounces = 0;
		}
		if (pathSeg->remainingBounces <= 0 && COMPACT) {
			indices[idx] = 0;
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

struct has_bounces
{
	__host__ __device__
	bool operator()(const PathSegment &a) {
		return a.remainingBounces > 0;
	}
};

/**
* Wrapper for the __global__ call that sets up the kernel calls and does a ton
* of memory management
*/
float pathtrace(uchar4 *pbo, int frame, int iter) {
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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths, dev_indices);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	startCpuTimer();
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if CACHE_FIRST	
		if (depth > 0) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth,
				num_paths,
				dev_paths,
				dev_indices,
				dev_sort_mat,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_intersections);
			checkCUDAError("trace one bounce");
		} else {
			if (iter == 1) {
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth,
					num_paths,
					dev_paths,
					dev_indices,
					dev_sort_mat,
					dev_geoms,
					hst_scene->geoms.size(),
					dev_intersections);
				checkCUDAError("trace one bounce");
				cudaMemcpy(dev_first_intersections, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			} else {
				cudaMemcpy(dev_intersections, dev_first_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			num_paths,
			dev_paths,
			dev_indices,
			dev_sort_mat,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_intersections);
		checkCUDAError("trace one bounce");
#endif

		cudaDeviceSynchronize();
		depth++;

#if CONTIG_MAT
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		thrust::sort_by_key(thrust::device, dev_sort_mat, dev_sort_mat + num_paths, dev_indices);
#endif

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_indices,
			dev_materials);

		if (depth >= traceDepth) {
			iterationComplete = true;
		} else {
#if COMPACT
		num_paths = StreamCompaction::Efficient_Shared::compact(num_paths, dev_indices, dev_indices);
		if (num_paths <= 0) {
			iterationComplete = true;
		}
#endif // end compact if
		}
	}
	endCpuTimer();

	num_paths = dev_path_end - dev_paths;

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
	return prev_elapsed_time_cpu_milliseconds;
}