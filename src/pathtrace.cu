#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_ptr.h>
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

#define ERRORCHECK 1

#define ANTIALIAS 1
#define CACHE_FIRST_BOUNCE 0 // Only works if ANTIALIAS is 0
#define SORT_MATERIALS 0

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

// Typedefs for sorting
typedef thrust::device_ptr<PathSegment> SegPtr;
typedef thrust::device_ptr<ShadeableIntersection> IntersectionPtr;
typedef thrust::tuple<SegPtr, IntersectionPtr> PtrTuple;
typedef thrust::zip_iterator<PtrTuple> ZipIterator;

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static SegPtr dev_thrust_paths;
static IntersectionPtr dev_thrust_intersections;
static int *dev_material_indices = nullptr;
static thrust::device_ptr<int> dev_thrust_material_indices;
static ShadeableIntersection *dev_first_bounce = NULL;
static bool first_bounce_cached = false;
static int * dev_light_indices = NULL;
static int num_lights;

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

  cudaMalloc(&dev_first_bounce, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));

  cudaMalloc(&dev_material_indices, pixelcount * sizeof(int));
  cudaMemset(dev_material_indices, -1, pixelcount * sizeof(int));

  dev_thrust_paths = SegPtr(dev_paths);
  dev_thrust_intersections = IntersectionPtr(dev_intersections);
  dev_thrust_material_indices = thrust::device_ptr<int>(dev_material_indices);

  // set up lights
  cudaMalloc(&dev_light_indices, scene->geoms.size() * sizeof(Geom));
  int light_index = 0;
  for (int i = 0; i < scene->geoms.size(); ++i) {
    if (scene->materials[scene->geoms[i].materialid].emittance > 0.0f) {
      //cudaMemcpy(&dev_light_indices[light_index], &dev_geoms[i], sizeof(Geom), cudaMemcpyDeviceToDevice);
      cudaMemset(&dev_light_indices[light_index], i, sizeof(int));
      ++light_index;
    }
    num_lights = light_index;
  }

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

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

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> u5(-0.5, 0.5);

		segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
#if ANTIALIAS
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u5(rng))
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u5(rng))
			);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__device__ void FindNearestIntersection(
    int geoms_size, Ray &pathSegRay, float& t_min, Geom *s_geoms,
    int &hit_geom_index, glm::vec3 &normal) {
	for (int i = 0; i < geoms_size; i++) {
	  float t;
	  glm::vec3 tmp_intersect;
	  glm::vec3 tmp_normal;
		Geom & geom = s_geoms[i];

		if (geom.type == CUBE) {
			t = boxIntersectionTest(geom, /*pathSegment.ray*/ pathSegRay, tmp_intersect, tmp_normal/*,
                              outside*/);
		} else if (geom.type == SPHERE) {
			t = sphereIntersectionTest(geom, /*pathSegment.ray*/ pathSegRay, tmp_intersect, tmp_normal/*,
                                 outside*/);
		}
		// TODO: add more intersection tests here... triangle? metaball? CSG?

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t) {
			t_min = t;
			hit_geom_index = i;
			//intersect_point = tmp_intersect;
			normal = tmp_normal;
		}
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	  int depth, int num_paths, PathSegment * pathSegments, Geom * geoms,
    int geoms_size, ShadeableIntersection * intersections) {
  extern __shared__ Geom s_geoms[];

	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x < geoms_size) {
    s_geoms[threadIdx.x] = geoms[threadIdx.x];
  }

  __syncthreads();

	if (path_index < num_paths) {
		//PathSegment pathSegment = pathSegments[path_index];
    Ray pathSegRay = pathSegments[path_index].ray;

		//glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		//bool outside = true;


		// naive parse through global geoms

    FindNearestIntersection(geoms_size, pathSegRay, t_min, s_geoms, 
                            hit_geom_index, normal);

		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;
		} else {
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = s_geoms[hit_geom_index].materialid;
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
    int iter, int num_paths, ShadeableIntersection * shadeableIntersections,
    PathSegment * pathSegments, Material * materials) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths && pathSegments[idx].remainingBounces > 0) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material &material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      PathSegment &seg = pathSegments[idx];
      if (material.emittance > 0.0f) {
        seg.color *= (materialColor * material.emittance);
        seg.remainingBounces = -1;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        /*float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not*/
        //PathSegment &seg = pathSegments[idx];
        if (material.hasReflective) {
          scatterRaySpecular(seg,
            seg.ray.origin + seg.ray.direction*intersection.t,
            intersection.surfaceNormal, material, rng);
        } else {
          scatterRayUniform(seg,
            seg.ray.origin + seg.ray.direction*intersection.t,
            intersection.surfaceNormal, material, rng);
        }
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

__global__ void DirectLighting(int num_paths,
                               PathSegment * pathSegments,
                               int *light_indices,
                               int lights_size,
                               Geom *geoms,
                               int geoms_size,
                               Material *materials,
                               int iter) {
  extern __shared__ Geom s_geoms[];

  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x < geoms_size) {
    s_geoms[threadIdx.x] = geoms[threadIdx.x];
  }
  __syncthreads();

  if (path_index < num_paths && pathSegments[path_index].remainingBounces >= 0) {
  //if (path_index < num_paths) {
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);
    int light_index = light_indices[int(u01(rng)*lights_size)];
    Geom &geom = s_geoms[light_index];
    glm::vec3 coord;
    if (geom.type == CUBE) {
      int dir = u01(rng) * 3;
      float sign = u01(rng) > 0.5 ? 1 : -1;
      glm::vec2 offset(u01(rng), u01(rng));
      switch (dir) {
      case 0:
        coord = glm::vec3(sign, offset.x, offset.y);
        break;
      case 1:
        coord = glm::vec3(offset.x, sign, offset.y);
        break;
      case 2:
        coord = glm::vec3(offset.x, offset.y, sign);
        break;
      }
    } else if (geom.type == SPHERE) {
      float t = u01(rng);
      float u = 2 * PI*u01(rng);
      float v = sqrt(t*(1 - t));
      coord = glm::vec3(2 * v*cos(u), 1 - 2 * t, 2 * v*sin(u));
    }
    glm::vec3 world_coord = multiplyMV(geom.transform, glm::vec4(coord, 1.0f));
    int hit_geom_index;
    {
      float t_min;
      glm::vec3 normal;
      Ray r = pathSegments[path_index].ray;
      r.direction = r.origin - world_coord;

      FindNearestIntersection(geoms_size, r, t_min,
        s_geoms, hit_geom_index, normal);
    }
    if (hit_geom_index == light_index) {
      pathSegments[path_index].color *= materials[geom.materialid].color * materials[geom.materialid].emittance;
    } else {
      pathSegments[path_index].color = glm::vec3(0.0f);
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

/*__global__ void is_terminated(int n, int *bools, ShadeableIntersection *inters) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < n) {
    bools[index] = inters[index].t < 0.0 ? 0 : 1;
  }
}*/
struct is_terminated {
  __device__ bool operator()(const PathSegment &seg) {
    return seg.color == glm::vec3(0.0f);
  }
};

//struct order_intersections {
//  __device__ bool operator()(const PtrTuple &a, const PtrTuple &b) {
//    return thrust::get<1>(a).get()->materialId <
//        thrust::get<1>(b).get()->materialId;
//  }
//};

__global__ void extract_material(int n, int *indices, const ShadeableIntersection *inters) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    indices[index] = inters->materialId;
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
  //num_paths = num_paths / 2-400;
  int num_curr_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	  // clean shading chunks
	  cudaMemset(dev_intersections, 0,
               pixelcount * sizeof(ShadeableIntersection));

	  // tracing
	  dim3 numblocksPathSegmentTracing =
        (num_curr_paths + blockSize1d - 1) / blockSize1d;
#if CACHE_FIRST_BOUNCE
    if (!first_bounce_cached && depth == 0) {
      computeIntersections<<<numblocksPathSegmentTracing, blockSize1d, sizeof(Geom) * hst_scene->geoms.size() + sizeof(Ray) * blockSize1d>>> (
          depth, num_curr_paths, dev_paths, dev_geoms, hst_scene->geoms.size(),
          dev_intersections);
      first_bounce_cached = true;
      cudaMemcpy(dev_first_bounce, dev_intersections, sizeof(ShadeableIntersection) * pixelcount, cudaMemcpyDeviceToDevice);
    } else if (depth != 0){
      computeIntersections<<<numblocksPathSegmentTracing, blockSize1d, sizeof(Geom) * hst_scene->geoms.size() + sizeof(Ray) * blockSize1d>>> (
          depth, num_curr_paths, dev_paths, dev_geoms, hst_scene->geoms.size(),
          dev_intersections);
    } else {
      cudaMemcpy(dev_intersections, dev_first_bounce, sizeof(ShadeableIntersection) * pixelcount, cudaMemcpyDeviceToDevice);
    }
#else
    computeIntersections<<<numblocksPathSegmentTracing, blockSize1d, sizeof(Geom) * hst_scene->geoms.size() + sizeof(Ray) * blockSize1d>>> (
        depth, num_curr_paths, dev_paths, dev_geoms, hst_scene->geoms.size(),
        dev_intersections);
#endif
	  checkCUDAError("trace one bounce");
	  cudaDeviceSynchronize();
	  depth++;
    printf("depth: %d\n", depth);


	  // TODO:
	  // --- Shading Stage ---
	  // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

    // SORT
#if SORT_MATERIALS
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(dev_thrust_paths, dev_thrust_intersections));
    extract_material<<<numblocksPathSegmentTracing, blockSize1d>>> (
        num_curr_paths, dev_material_indices, dev_intersections);
    thrust::sort_by_key(dev_thrust_material_indices,
                        dev_thrust_material_indices + num_curr_paths,
                        zipped);
#endif

    shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
      iter,
      num_curr_paths,
      dev_intersections,
      dev_paths,
      dev_materials
    );
    //PathSegment *seg = (PathSegment*)malloc(sizeof(PathSegment)*num_paths);
    //cudaMemcpy(seg, dev_paths, sizeof(PathSegment)*num_paths, cudaMemcpyDeviceToHost);
    //printf("remaining: %d\n", seg[0].remainingBounces);

    thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
    thrust::device_ptr<PathSegment> new_end = thrust::remove_if(
        thrust_paths,
        thrust_paths + num_curr_paths,
        is_terminated());
    num_curr_paths = new_end - thrust_paths;

    //iterationComplete = num_curr_paths == 0 ? true : false; // TODO: should be based off stream compaction results.

    //PathSegment *seg2 = (PathSegment*)malloc(sizeof(PathSegment)*num_paths);
    //cudaMemcpy(seg2, dev_paths, sizeof(PathSegment)*num_paths, cudaMemcpyDeviceToHost);
    //printf("remaining: %d\n", seg2[0].remainingBounces);
    //free(seg);
    //free(seg2);
    printf("paths: %d\n", num_curr_paths);
    iterationComplete = depth >= traceDepth ? true : false;
	}
	dim3 numblocksPathSegmentTracing =
      (num_curr_paths + blockSize1d - 1) / blockSize1d;
  DirectLighting<<<numblocksPathSegmentTracing, blockSize1d, sizeof(Geom) * hst_scene->geoms.size()>>>(
      num_curr_paths, dev_paths, dev_light_indices, num_lights, dev_geoms, hst_scene->geoms.size(),
      dev_materials, iter);

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_curr_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
