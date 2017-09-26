#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>


#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "stream_compaction/EfficientStreamCompaction.h"

#define ERRORCHECK 1

// #define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// #define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
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
static PathSegment * dev_first_bounce_paths = NULL;

static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_first_bounce_intersections = NULL;


//static int* radix_sort_b_array;
//static int* radix_sort_e_array;
//static int* radix_sort_f_array;
//
//static int* radix_sort_host_f_array;
//
//static PathSegment * dev_paths_after_sort = NULL;
//static ShadeableIntersection * dev_intersections_after_sort = NULL;


// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));
	//cudaMalloc(&dev_paths_after_sort, pixelcount * sizeof(PathSegment));



  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	//cudaMalloc(&dev_intersections_after_sort, pixelcount * sizeof(ShadeableIntersection));
	//cudaMemset(dev_intersections_after_sort, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_bounce_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	/*cudaMalloc((void**)&radix_sort_b_array, pixelcount * sizeof(int));
	cudaMalloc((void**)&radix_sort_e_array, pixelcount * sizeof(int));
	cudaMalloc((void**)&radix_sort_f_array, pixelcount * sizeof(int));

	radix_sort_host_f_array = new int[pixelcount];*/

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null

  	cudaFree(dev_paths);
	cudaFree(dev_first_bounce_paths);
	//cudaFree(dev_paths_after_sort);

  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);

  	cudaFree(dev_intersections);
	cudaFree(dev_first_bounce_intersections);
	//cudaFree(dev_intersections_after_sort);
    // TODO: clean up any extra device memory you created

	/*cudaFree(radix_sort_b_array);
	cudaFree(radix_sort_e_array);
	cudaFree(radix_sort_f_array);

	delete[] radix_sort_host_f_array;*/

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

			// we need sort by this, assign -1 for those have no intersections
			//intersections[path_index].materialTypeID = -1;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;

			//intersections[path_index].materialTypeID = geoms[hit_geom_index].materialTypeID;
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

// First Version of Shade function
__global__ void shadeMaterialNaive(
	int iter
	, int num_paths
	, int depth
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& pathSegment = pathSegments[idx];

		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;


			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegment.color *= (materialColor * material.emittance);
				// Terminate the ray if it hist a light
				pathSegment.remainingBounces = 0;
			}
			
			else {
				
				scatterRay(pathSegment, 
						   getPointOnRay(pathSegment.ray, intersection.t),
						   intersection.surfaceNormal,
					       material,
						   rng);

				pathSegment.remainingBounces--;
			}

			
		}
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
		else {
			pathSegment.color = glm::vec3(0.0f);

			pathSegment.remainingBounces = 0;
		}
	}
}




// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths, glm::vec3 ambientColor)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += (iterationPath.color + ambientColor);
	}
}


// ----------------- Radix Sort ------------------------
__global__ void kernGen_b_e_array(int N, int idxBit, int* b_array, int* e_array, const ShadeableIntersection *dev_data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	int temp_result = (dev_data[index].materialId >> idxBit) & 1;
	b_array[index] = temp_result;
	e_array[index] = 1 - temp_result;
}

__global__ void kern_Gen_d_array_and_scatter(int N, const int totalFalses, const int* b_array, const int* f_array, 
	ShadeableIntersection* dev_odata1, ShadeableIntersection* dev_idata1,
	PathSegment* dev_odata2, PathSegment* dev_idata2)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	int t_array_value = index - f_array[index] + totalFalses;

	int d_array_value = b_array[index] ? t_array_value : f_array[index];

	dev_odata1[d_array_value] = dev_idata1[index];

	dev_odata2[d_array_value] = dev_idata2[index];
}

// ------------------------------------------------------


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
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// Cache camera rays the very first sample per pixel
	// and the intersections of these rays
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_first_bounce_paths);
		checkCUDAError("generate camera ray");

		// clean shading chunks
		cudaMemset(dev_first_bounce_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		computeIntersections << <dim3((pixelcount + blockSize1d - 1) / blockSize1d), blockSize1d >> > (
			depth
			, num_paths
			, dev_first_bounce_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_first_bounce_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
	}

	// Directly copy first camera rays and their intersections from the very first sample per pixel
	cudaMemcpy(dev_paths, dev_first_bounce_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_intersections, dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);


	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
	while (!iterationComplete) {

		// tracing
		// num_paths : number of paths/pathSegments that we should still trace
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		// Do not compute intersections when depth is 0,
		// Already copy it from dev_first_bounce_intersections
		if (depth) {
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

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
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	    // evaluating the BSDF.
	    // Start off with just a big kernel that handles all the different
	    // materials you have in the scenefile.

	    // TODO: compare between directly shading the path segments and shading
	    // path segments that have been reshuffled to be contiguous in memory.

		// Sort here by ShadeableIntersection's material ID
		//thrust::device_ptr<ShadeableIntersection> dev_thrust_keys(dev_intersections);
		//thrust::device_ptr<PathSegment> dev_thrust_values(dev_paths);
		//thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + num_paths, dev_thrust_values, thrust::greater<ShadeableIntersection>());


		
		// Radix Sort
		// 4 kinds of materials, we temporarily set numbOfBits to 3
		//int numOfBits = 3;
		//for (int k = 0; k <= numOfBits - 1; k++) {
		//	kernGen_b_e_array << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, k, radix_sort_b_array, radix_sort_e_array, dev_intersections);

		//	cudaMemcpy(radix_sort_host_f_array, radix_sort_e_array, sizeof(int) * num_paths, cudaMemcpyDeviceToHost);

		//	// totalFalses = e_array[n-1]
		//	int totalFalses = radix_sort_host_f_array[num_paths - 1];

		//	// Get Exclusive scan result as a whole
		//	StreamCompaction::Efficient::scanDynamicShared(num_paths, radix_sort_host_f_array, radix_sort_host_f_array);

		//	// totalFalses += f_array[n-1]
		//	totalFalses += radix_sort_host_f_array[num_paths - 1];

		//	cudaMemcpy(radix_sort_f_array, radix_sort_host_f_array, sizeof(int) * num_paths, cudaMemcpyHostToDevice);

		//	// Since here we run exclusive scan as a whole,
		//	// and we don't want each tile to run StreamCompaction::Efficient::scan individually.
		//	// value in d_array here is actually index value in the whole data array, not just index in that tile
		//	// so, there is NO need to merge here
		//	kern_Gen_d_array_and_scatter<< <numblocksPathSegmentTracing, blockSize1d >> > 
		//		(num_paths, totalFalses, 
		//		radix_sort_b_array, radix_sort_f_array, 
		//		dev_intersections_after_sort, dev_intersections, 
		//		dev_paths_after_sort, dev_path s);

		//	ShadeableIntersection* temp1 = dev_intersections;
		//	dev_intersections = dev_intersections_after_sort;
		//	dev_intersections_after_sort = temp1;
		//	
		//	PathSegment* temp2 = dev_paths;
		//	dev_paths = dev_paths_after_sort;
		//	dev_paths_after_sort = temp2;
		//}



	    /*shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
		  iter,
		  num_paths,
		  dev_intersections,
		  dev_paths,
		  dev_materials
	    );*/

		shadeMaterialNaive<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			depth,
			dev_intersections,
			dev_paths,
			dev_materials
		);

		
		
		// actually, for the last path tracing, there is not need to 
		// conduct stream compaction, because there is no following path-tracing
		if (depth == traceDepth) {
			iterationComplete = true; 
			continue;
		}

		// num_paths : number of paths/pathSegments that we should still trace
		// iterationComplete should be based off stream compaction results
		num_paths = StreamCompaction::Efficient::compactDynamicShared(num_paths, dev_paths);

		// if there is no paths meet the requirement(remaining bounces > 0), we just end the loop
		if (num_paths == 0) {
			iterationComplete = true;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

	glm::vec3 ambientColor = glm::vec3(0.1f, 0.1f, 0.1f);

	//finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
	finalGather << <numBlocksPixels, blockSize1d >> >(pixelcount, dev_image, dev_paths, ambientColor);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
