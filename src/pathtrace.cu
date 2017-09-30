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

//Include Stream Compaction files
#include "stream_compaction\efficient.h"


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

// =============================================================================
//					PATH TRACE INIT AND FREE CPU FUNCTIONS
// =============================================================================

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

//PathSeg and Isect indices sorted by Material
static int * dev_PathSegIndices = NULL;
static int * dev_IsectIndices = NULL;

//Caching first bounce
static ShadeableIntersection * dev_IsectCached = NULL;

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

	//PathSeg and Isect indices sorted by Material
	cudaMalloc(&dev_PathSegIndices, pixelcount * sizeof(int));

	cudaMalloc(&dev_IsectIndices, pixelcount * sizeof(int));

	//Caching first bounce
	cudaMalloc(&dev_IsectCached, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_IsectCached, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

	cudaFree(dev_PathSegIndices);
	cudaFree(dev_IsectIndices);
	cudaFree(dev_IsectCached);

    checkCUDAError("pathtraceFree");
}

// =============================================================================
//				GENERATE 3D POINT BASED ON DISK CONOCENTRIC
// =============================================================================

__host__ __device__
glm::vec3 squareToDiskConocentric(const glm::vec2 &sample)
{
	glm::vec3 output(0.0f);
	glm::vec2 uOffset = 2.0f * sample - glm::vec2(1.0f);

	if (uOffset.x == 0.0f + RAY_EPSILON && uOffset.y == 0.0f + RAY_EPSILON)
	{
		return glm::vec3(0.0f);
	}

	float theta = 0.0f;
	float r = 0.0f;

	if (std::fabs(uOffset.x) > std::fabs(uOffset.y))
	{
		r = uOffset.x;
		theta = (PI / 4.0f) *  (uOffset.y / uOffset.x);
	}

	else
	{
		r = uOffset.y;
		theta = (PI / 2.0f) - (PI / 4.0f) * (uOffset.x / uOffset.y);
	}

	output = r * glm::vec3(cos(theta), sin(theta), 0.0f);
	return output;
}


// =============================================================================
//					GENERATE RAY FROM CAMERA KERNEL FUNCTION
// =============================================================================

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

	if (x < cam.resolution.x && y < cam.resolution.y) 
	{
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		// Note: If antialiasing -- can NOT cache first bounce!

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
		thrust::uniform_real_distribution<float> u01(-1, 1);
		float offset_x = u01(rng);
		float offset_y = u01(rng);

		//Non anti-aliased
		segment.ray.direction = glm::normalize(cam.view
												- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
												- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
												);
		
		//Anti-Aliasing
		//segment.ray.direction = glm::normalize(cam.view
		//										- cam.right * cam.pixelLength.x * ((float)x + offset_x - (float)cam.resolution.x * 0.5f)
		//										- cam.up * cam.pixelLength.y * ((float)y + offset_y - (float)cam.resolution.y * 0.5f)
		//										);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		//Depth of field
		//Generate sample point on disk lens, shoot ray through that
		if (cam.lensRadius > 0.0f)
		{
			thrust::uniform_real_distribution<float> u02(0, 1);
			glm::vec2 sample = glm::vec2(u02(rng), u02(rng));

			glm::vec3 pLens = cam.lensRadius * squareToDiskConocentric(sample);
			glm::vec3 pFocus = cam.focalDistance * segment.ray.direction + segment.ray.origin;
			glm::vec3 aperaturePt = segment.ray.origin + (cam.up * pLens[1]) + (cam.right * pLens[0]);

			segment.ray.origin = aperaturePt;
			segment.ray.direction = glm::normalize(pFocus - aperaturePt);
		}//end DOF check

	}//end if 
}

// =============================================================================
//						COMPUTE INTERSECTION KERNEL FUNCTION
// =============================================================================

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
		}//end for all geoms

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;			//QUESTION: should this be normalized?
			//intersections[path_index].surfaceNormal = glm::normalize(normal);
			intersections[path_index].intersectionPt = intersect_point;
		}
	}
}


// =============================================================================
//							SHADER KERNEL FUNCTIONS
// =============================================================================


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
	, int depth
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces <= 0) 
		{
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		
		// if the intersection exists...
		if (intersection.t > 0.0f) 
		{ 
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at makeSeededRandomEngine as well.

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			
			//Use depth for non-compact version to work properly
			//https://groups.google.com/forum/#!topic/cis-565-fall-2017/thgdf2jzDyo
			//thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			//thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);


			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) 
			{
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else 
			{
				//float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				//pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				//pathSegments[idx].color *= u01(rng); // apply some noise because why not

				scatterRay(pathSegments[idx], intersection.intersectionPt, intersection.surfaceNormal, material, rng);
				pathSegments[idx].remainingBounces--;

			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}//end if

		else 
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}//end else
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



// =============================================================================
//							PATH TERMINATION SCAN KERNEL
// =============================================================================

// Check if remaining bounces == 0
// Check if path intersection t value == -1 (didn't hit anything)

//UNUSED
__global__ void kernMapRemainingBouncesToBoolean(int n, int *bools, PathSegment *pathSegments)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n)
	{
		//If path's remanining bounces is > 0, mark as 1, else 0
		PathSegment currPath = pathSegments[index];
		if (currPath.remainingBounces > 0)		bools[index] = 1;
		else									bools[index] = 0;
	}
}//end kernMapRemainingBounces

//UNUSED
__global__ void kernMapNoIsectPathToBoolean(int n, int *bools, ShadeableIntersection *intersections)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n)
	{
		ShadeableIntersection currIsect = intersections[index];
		//if()
	}
}

// Predicate for thrust::partition
struct hasRemainingBounces
{
	__host__ __device__
	bool operator()(const PathSegment &pathSegment)
	{
		return pathSegment.remainingBounces > 0;
	}
};


//Fill dev_PathSegIndices and dev_IsectIndices with their corresponding material ID
// These arrays should essentially be the same since pathSeg's and Isects correspond to each other
__global__ void kernSortByMaterial(int n, int *pathSegIndices, int *isectIndices, ShadeableIntersection *isects)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n)
	{
		int currMatID = isects[index].materialId;
		pathSegIndices[index] = currMatID;
		isectIndices[index] = currMatID;
	}
}

// =============================================================================
//							PATH TRACING CPU FUNCTION
// =============================================================================

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) 
{
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

	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	int num_remainingPaths = num_paths;


	//NOTE: USE TIMER TO PRINT THE TIME FOR EACH ITERATION OF WHILE LOOP


	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) 
	{
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		//DO I NEED THIS? OR DO I NOT WANT TO DO THIS BECAUSE I WANT TO KEEP USING IT?
		//cudaMemset(dev_IsectCached, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		//dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		dim3 numblocksPathSegmentTracing = (num_remainingPaths + blockSize1d - 1) / blockSize1d;

		//Caching first bounce
		//Don't start at iter = 0, that's ray from camera to screen
		
		// Compute intersections

		if (CACHE_FIRST_BOUNCE && depth == 0 && iter == 1)
		{
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
																				depth
																				, num_remainingPaths //num_paths
																				, dev_paths
																				, dev_geoms
																				, hst_scene->geoms.size()
																				, dev_IsectCached
																				);
		}//end if
		else
		{
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
																				depth
																				, num_remainingPaths //num_paths
																				, dev_paths
																				, dev_geoms
																				, hst_scene->geoms.size()
																				, dev_intersections
																				);
		}//end else

		//TESTING 
		//computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
		//	depth
		//	, num_remainingPaths //num_paths
		//	, dev_paths
		//	, dev_geoms
		//	, hst_scene->geoms.size()
		//	, dev_intersections
		//	);


		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		//depth++;


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		if (CACHE_FIRST_BOUNCE && depth == 0 && iter == 1)
		{
			if (SORT_BY_MATERIAL)
			{
				//Store material ID's in dev_PathSegIndices and dev_IsectIndices respectively
				kernSortByMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(num_remainingPaths, dev_PathSegIndices, dev_IsectIndices, dev_IsectCached);

				//Sort the PathSegments and Isects arrays in place according to materialID's placed in their corresponding dev_indices arrays 
				thrust::sort_by_key(thrust::device, dev_PathSegIndices, dev_PathSegIndices + num_remainingPaths, dev_paths);
				thrust::sort_by_key(thrust::device, dev_IsectIndices, dev_IsectIndices + num_remainingPaths, dev_IsectCached);
			}

			shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
																			iter,
																			num_remainingPaths, //num_paths,
																			dev_IsectCached,
																			dev_paths,
																			dev_materials,
																			depth
																			);
		}//end if first bounce

		//Operating on everything else after first bounce
		else
		{
			if (SORT_BY_MATERIAL)
			{
				//Store material ID's in dev_PathSegIndices and dev_IsectIndices respectively
				kernSortByMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(num_remainingPaths, dev_PathSegIndices, dev_IsectIndices, dev_intersections);

				//Sort the PathSegments and Isects arrays in place according to materialID's placed in their corresponding dev_indices arrays 
				thrust::sort_by_key(thrust::device, dev_PathSegIndices, dev_PathSegIndices + num_remainingPaths, dev_paths);
				thrust::sort_by_key(thrust::device, dev_IsectIndices, dev_IsectIndices + num_remainingPaths, dev_intersections);
			}


			shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
																			iter,
																			num_remainingPaths, //num_paths,
																			dev_intersections,
																			dev_paths,
																			dev_materials,
																			depth
																			);
		}//end else 

		//TESTING
		//shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
		//																iter,
		//																num_remainingPaths, //num_paths,
		//																dev_intersections,
		//																dev_paths,
		//																dev_materials,
		//																depth
		//																);


		// TODO: Stream compact away all of the terminated paths.
		// You may use either your implementation or `thrust::remove_if` or its cousins.
		// NOTE: DO NOT USE REMOVE_IF
		PathSegment* lastRemainingPath = thrust::partition(thrust::device, dev_paths, dev_paths + num_remainingPaths, hasRemainingBounces());
		num_remainingPaths = lastRemainingPath - dev_paths;

		// Increment depth
		depth++;


		// TODO: should be based off stream compaction results.
		// To test anti-aliasing, change depth >= 1, and move the camera around. You'll see jagged edges become smoother
		iterationComplete = ((depth >= traceDepth || num_remainingPaths <= 0) ? true : false);

	}//end while

	  // Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
				pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}//end pathTrace
