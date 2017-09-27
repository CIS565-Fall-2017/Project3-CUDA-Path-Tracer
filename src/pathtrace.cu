#include <cstdio>
#include <cuda.h>
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "stream_compaction/common.h"
#include "stream_compaction/efficient.h"

#define ERRORCHECK 1

// Toggle features
#define PATH_COMPACT 1
#define CACHE_FIRST_BOUNCE 1
#define SORT_BY_MATERIAL 0
#define STOCHASTIC_ANTIALIASING 0
#define DEPTH_OF_FIELD 0
#define MOTION_BLUR 0

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

using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer& timer()
{
	static PerformanceTimer timer;
	return timer;
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
static ShadeableIntersection * dev_intersections_firstbounce = NULL;
static int * dev_materials_paths = NULL;
static int * dev_materials_intersections = NULL;

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

    cudaMalloc(&dev_intersections_firstbounce, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_firstbounce, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_materials_paths, pixelcount * sizeof(int));
    cudaMalloc(&dev_materials_intersections, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    cudaFree(dev_intersections_firstbounce);
    cudaFree(dev_materials_paths);
    cudaFree(dev_materials_intersections);
	
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

		// Stochastic sampled antialiasing implemented by jittering the ray
#if STOCHASTIC_ANTIALIASING
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
    thrust::uniform_real_distribution<float> u01(0, 1);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng) - 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng) - 0.5f)
			);
#else
    segment.ray.direction = glm::normalize(cam.view
      - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
      - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
    );
#endif

    // Depth of Field
#if DEPTH_OF_FIELD
      thrust::default_random_engine rngx = makeSeededRandomEngine(iter, x, 0);
      thrust::default_random_engine rngy = makeSeededRandomEngine(iter, y, 0);
      thrust::uniform_real_distribution<float> udof(-1.0, 1.0);

      float lensU; float lensV;
      lensU = (udof(rngx)) / (80.0f);
      lensV = (udof(rngy)) / (80.0f);

      float t = 2.0f;
      glm::vec3 focus = t * segment.ray.direction + segment.ray.origin;
      segment.ray.origin += cam.right * lensU + cam.up * lensV;
      segment.ray.direction = glm::normalize(focus - segment.ray.origin);
#endif

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
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].point = intersect_point;
		}
	}
}

__global__ void shadeMaterial(
  int iter
  , int num_paths
  , ShadeableIntersection * shadeableIntersections
  , PathSegment * pathSegments
  , Material * materials
  , int depth
  , glm::vec3* img
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  PathSegment& path_segment = pathSegments[idx];
  if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
                                 // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        path_segment.color *= (materialColor * material.emittance);
        path_segment.remainingBounces = 0;
      }
      else {
        scatterRay(path_segment, intersection.point, intersection.surfaceNormal, material, rng);
        path_segment.remainingBounces--;
      }
      // If there was no intersection, color the ray black.
      // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
      // used for opacity, in which case they can indicate "no opacity".
      // This can be useful for post-processing and image compositing.
    }
    else {
      path_segment.color = glm::vec3(0.0f);
      path_segment.remainingBounces = 0;
    }

    if (path_segment.isDone()) {
      img[path_segment.pixelIndex] += path_segment.color;
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

struct isPathInactive {
  __host__ __device__ bool operator() (const PathSegment& path_segment) {
    return (path_segment.remainingBounces <= 0);
  }
};

__global__ void kernGetMaterialId(int n, int *odata, const ShadeableIntersection *intersections) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index > n)
  {
    odata[index] = intersections[index].materialId;
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	//timer().startGpuTimer();
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

	const Geom *hst_scene_geoms = &(hst_scene->geoms)[0];
	Geom *motionBlurGeoms = &(hst_scene->geoms)[0];

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


#if MOTION_BLUR
  thrust::default_random_engine rng = makeSeededRandomEngine(iter, hst_scene->geoms.size(), traceDepth);
  thrust::uniform_real_distribution<float> umotion(0, TWO_PI);

  for (int i = 0; i < hst_scene->geoms.size(); i++) {
    Geom& currGeom = motionBlurGeoms[i];
    currGeom = hst_scene_geoms[i];

    currGeom.translation.x += hst_scene_geoms[i].motion.x * 0.06 * cos(umotion(rng));
    currGeom.translation.y += hst_scene_geoms[i].motion.y * 0.06 * cos(umotion(rng));
    currGeom.translation.z += hst_scene_geoms[i].motion.z * 0.06 * cos(umotion(rng));

    // calculate transforms of geometry
    currGeom.transform = utilityCore::buildTransformationMatrix(
      currGeom.translation, currGeom.rotation, currGeom.scale);
    currGeom.inverseTransform = glm::inverse(currGeom.transform);
    currGeom.invTranspose = glm::inverseTranspose(currGeom.transform);
  }

  cudaMemcpy(dev_geoms, motionBlurGeoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

  generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
  checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
  int num_paths_active = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	  // clean shading chunks
	  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	  // tracing
	  dim3 numblocksPathSegmentTracing = (num_paths_active + blockSize1d - 1) / blockSize1d;
    if (!CACHE_FIRST_BOUNCE || (CACHE_FIRST_BOUNCE && ((depth > 0) || (depth == 0 && iter == 1)))) {
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

#if CACHE_FIRST_BOUNCE
	if (depth == 0 && iter == 1) {
		cudaMemcpy(dev_intersections_firstbounce, dev_intersections, num_paths_active * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	}
	// re-use first bounce across all subsequent iterations
	else if (depth == 0 && iter > 1) {
		cudaMemcpy(dev_intersections, dev_intersections_firstbounce, num_paths_active * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	}
#endif

	  // TODO:
	  // --- Shading Stage ---
	  // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

#if SORT_BY_MATERIAL
    kernGetMaterialId << <numblocksPathSegmentTracing, blockSize1d >> >(num_paths_active, dev_materials_paths, dev_intersections);
    checkCUDAError("kernGetMaterialType failed");

    cudaMemcpy(dev_materials_intersections, dev_materials_paths, num_paths_active * sizeof(int), cudaMemcpyDeviceToDevice);

    thrust::sort_by_key(thrust::device, dev_materials_paths, dev_materials_paths + num_paths_active, dev_paths);
    thrust::sort_by_key(thrust::device, dev_materials_intersections, dev_materials_intersections + num_paths_active, dev_intersections);
#endif

    shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
      iter,
      num_paths_active,
      dev_intersections,
      dev_paths,
      dev_materials,
      depth,
      dev_image
      );

#if PATH_COMPACT
    // Thrust compact
    thrust::device_ptr<PathSegment> thrust_dev_paths_ptr(dev_paths);
    auto thrust_end = thrust::remove_if(thrust::device, thrust_dev_paths_ptr, thrust_dev_paths_ptr + num_paths_active, isPathInactive());
    num_paths_active = thrust_end - thrust_dev_paths_ptr;
#endif

    depth++;
    iterationComplete = num_paths_active <= 0 || traceDepth < depth; // TODO: should be based off stream compaction results.
	}

	/*timer().endGpuTimer();
	if (iter < 101 && (iter % 10 == 0 || iter == 1)) {
		cout << timer().getGpuElapsedTimeForPreviousOperation() << endl;
	}*/

  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
          pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}
