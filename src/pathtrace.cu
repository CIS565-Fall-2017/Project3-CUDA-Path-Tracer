#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

// Defines for toggling features
#define COMPACTION 0
#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_HIT 1
#define DIRECT_LIGHTING 1

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

// Predicate for removing terminated paths
struct is_terminated_path {
  __host__ __device__
    bool operator()(const PathSegment &pathSegment) {
    return pathSegment.remainingBounces > 0;
  }
};

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
#if SORT_BY_MATERIAL
static int *dev_materialIds1 = NULL;
static int *dev_materialIds2 = NULL;
#endif

#if CACHE_FIRST_HIT
static int hasCacheBeenMade = 0;
static ShadeableIntersection *dev_intersectionCache = NULL;
#endif

#if DIRECT_LIGHTING
static int num_lights = 0;
static int *dev_lightIdxs = NULL;
#endif

static Mesh *dev_meshes = NULL;
static Triangle *dev_tris = NULL;

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
#if SORT_BY_MATERIAL
    cudaMalloc(&dev_materialIds1, pixelcount * sizeof(int));
    cudaMalloc(&dev_materialIds2, pixelcount * sizeof(int));
#endif

#if CACHE_FIRST_HIT
    hasCacheBeenMade = 0;
    cudaMalloc(&dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection));
#endif

#if DIRECT_LIGHTING
    std::vector<int> hst_temp_lightIdxs = std::vector<int>();
    for (int i = 0; i < scene->geoms.size(); ++i) {
      Geom geom = scene->geoms[i];
      if (scene->materials[geom.materialid].emittance > 0.0f) {
        hst_temp_lightIdxs.push_back(i);
      }
    }
    num_lights = hst_temp_lightIdxs.size();
    cudaMalloc(&dev_lightIdxs, scene->geoms.size() * sizeof(int));
    checkCUDAError("pre-copy");
    if (num_lights > 0) {
      cudaMemcpy(dev_lightIdxs, hst_temp_lightIdxs.data(), num_lights * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("copy");
    }
#endif

    cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Mesh));
    cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
    cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
#if SORT_BY_MATERIAL
    cudaFree(dev_materialIds1);
    cudaFree(dev_materialIds2);
#endif

#if CACHE_FIRST_HIT
    cudaFree(dev_intersectionCache);
#endif

#if DIRECT_LIGHTING
    cudaFree(dev_lightIdxs);
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

/* Returns hit_geom_index */
__device__ int naiveGeomParse(Geom *geoms, int geoms_size, Ray &ray, Mesh *meshes, Triangle *tris,
                              glm::vec3 &intersect_point, glm::vec3 &normal, float &t_min) {

  t_min = FLT_MAX;
  float t;
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
    else if (geom.type == MESH)
    {
      t = meshIntersectionTest(geom, ray, meshes, tris, tmp_intersect, tmp_normal);
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

  return hit_geom_index;
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
  , Mesh *meshes
  , Triangle *tris
  , int *materialIds1 = NULL
  , int *materialIds2 = NULL
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

#if 0
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
#else

    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min;

    int hit_geom_index = naiveGeomParse(geoms, geoms_size, pathSegment.ray, meshes, tris,
      intersect_point, normal, t_min);
#endif

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
#if SORT_BY_MATERIAL
      materialIds1[path_index] = -1;
      materialIds2[path_index] = -1;
#endif
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
      int materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].materialId = materialId;
#if SORT_BY_MATERIAL
      materialIds1[path_index] = materialId;
      materialIds2[path_index] = materialId;
#endif
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
#if DIRECT_LIGHTING
  , Geom *geoms
  , int geoms_size
  , int *lightIdxs
  , int num_lights
  , Mesh *meshes
  , Triangle *tris
#endif
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
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      if (pathSegments[idx].remainingBounces <= 0) {
        return;
      }

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);

        pathSegments[idx].remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {

        PathSegment &pathSegment = pathSegments[idx];
        scatterRay(pathSegments[idx],
          pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t,
          intersection.surfaceNormal,
          material,
          rng);
        pathSegments[idx].remainingBounces--;

#if DIRECT_LIGHTING
        if (pathSegments[idx].remainingBounces == 0) {
          thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
          thrust::uniform_real_distribution<float> u01(0.0f, 0.99999f);
          int randomIdx = (int)floor(u01(rng) * num_lights);
          int lightIdx = lightIdxs[randomIdx];
          // ignore geometry type for now
          glm::vec4 samplePoint = glm::vec4(u01(rng), u01(rng), u01(rng), 1);
          samplePoint = geoms[lightIdx].transform * samplePoint;
          Ray sampleRay = pathSegments[idx].ray;
          sampleRay.direction = glm::normalize(glm::vec3(samplePoint) - sampleRay.origin);
          glm::vec3 dummyVec;
          float dummyFloat;
          int hit_geom_index = naiveGeomParse(geoms, geoms_size, sampleRay, meshes, tris,
            dummyVec, dummyVec, dummyFloat);
          if (hit_geom_index == lightIdx) {
            Material lightMaterial = materials[geoms[lightIdx].materialid];
            pathSegments[idx].color *= lightMaterial.color * lightMaterial.emittance;
          }
        }
#endif
      }
      // If there was no intersection, color the ray black.
      // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
      // used for opacity, in which case they can indicate "no opacity".
      // This can be useful for post-processing and image compositing.
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

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_HIT
  if (depth != 0 || !hasCacheBeenMade) {
#else
  if (1) {
#endif
    // tracing

    computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
      depth
      , num_paths
      , dev_paths
      , dev_geoms
      , hst_scene->geoms.size()
      , dev_intersections
      , dev_meshes
      , dev_tris
#if SORT_BY_MATERIAL
      , dev_materialIds1
      , dev_materialIds2
#endif
      );
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
#if CACHE_FIRST_HIT
    if (depth == 0) {
      cudaMemcpy(dev_intersectionCache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
      hasCacheBeenMade = 1;
    }
  }
  else {
    cudaMemcpy(dev_intersections, dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
#endif
  }

	depth++;
#if SORT_BY_MATERIAL
  thrust::sort_by_key(thrust::device, dev_materialIds1, dev_materialIds1 + num_paths, dev_paths);
  thrust::sort_by_key(thrust::device, dev_materialIds2, dev_materialIds2 + num_paths, dev_intersections);
#endif
	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

  shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
#if DIRECT_LIGHTING
    ,
    dev_geoms,
    hst_scene->geoms.size(),
    dev_lightIdxs,
    num_lights,
    dev_meshes,
    dev_tris
#endif
  );
#if COMPACTION
  dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_terminated_path());
  num_paths = dev_path_end - dev_paths;
  iterationComplete = (num_paths <= 0); // TODO: should be based off stream compaction results.
#else 
  iterationComplete = (depth >= traceDepth);
#endif
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
