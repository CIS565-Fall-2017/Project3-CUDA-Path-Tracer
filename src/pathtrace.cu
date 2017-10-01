#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <glm/gtc/matrix_inverse.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
using namespace std;

#define ERRORCHECK 1
#define ANTIALIASING 1
#define STREAM_COMPACTION 0 // very slow
#define CACHE_FIRST_BOUNCE 0 // set 0 when using DOF || ANTIALIASING
#define SORT_MATERIAL 0 // extremely slow
#define STRATIFIED 0
#define DOF 0
#define MSI 0
#define DIRECT_LIGHTING 0
#define TIMER 0
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
static Geom * dev_lights = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

static ShadeableIntersection * dev_first_intersections = NULL;
static bool cache_first_bounce = false;
//static int * sqrtival_x = NULL;
//static int * sqrtival_y = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

	cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	/*const int sqrtVal = (int)(std::sqrt((float)scene->state.iterations) + 0.5);
	int * temp = (int*)malloc(sqrtVal * sizeof(int));

	for (int i = 0; i < sqrtVal; i++)
	{
		temp[i] = i;
	}

	random_shuffle(temp, temp + sqrtVal);

	cudaMalloc(&sqrtival_x, sqrtVal * sizeof(int));
	cudaMemcpy(sqrtival_x, temp, sqrtVal * sizeof(int), cudaMemcpyHostToDevice);

	random_shuffle(temp, temp + sqrtVal);

	cudaMalloc(&sqrtival_y, sqrtVal * sizeof(int));
	cudaMemcpy(sqrtival_x, temp, sqrtVal * sizeof(int), cudaMemcpyHostToDevice);

	cache_first_bounce = false;*/

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
	cudaFree(dev_lights);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

	cudaFree(dev_first_intersections);
	//free(sqrtival_x);
	//free(sqrtival_y);
	//cudaFree(sqrtival_x);
	//cudaFree(sqrtival_y);

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
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
#if ANTIALIASING
		// TODO: implement antialiasing by jittering the ray
		thrust::uniform_real_distribution<float> uAA(-0.5, 0.5);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + uAA(rng))
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + uAA(rng))
		);
#else

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
//Reference: Physically Based Rendering, Third Edition p374
#if DOF
		thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
		glm::vec2 pLens = cam.lensRadius * squareToDiskConcentric(sample);
		float ft = glm::abs(cam.focalDistance / segment.ray.direction.z);
		glm::vec3 pFocus = segment.ray.origin + segment.ray.direction * ft;
		segment.ray.origin += pLens.x * cam.right + pLens.y * cam.up;
		segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
#else
#endif
		

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}
__device__ void rayIntersections(
	Ray * ray
	, Geom * geoms
	, int geoms_size
	, float & t_min
	, int & hit_geom_index
)
{
	float t;
	bool outside = true;
	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// naive parse through global geoms

	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, *ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, *ray, tmp_intersect, tmp_normal, outside);
		}
		// TODO: add more intersection tests here... triangle? metaball? CSG?

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t)
		{
			hit_geom_index = i;
			t_min = t;
		}
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
	, int num_lights
	, int num_geoms
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, Geom * lights
	, Geom * geoms
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths && pathSegments[idx].remainingBounces)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.


      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
		pathSegments[idx].remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
		  thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
#if STRATIFIED
		  thrust::default_random_engine rng2 = makeSeededRandomEngine(iter, idx, -pathSegments[idx].remainingBounces);
		  thrust::uniform_real_distribution<float> u01(0, 1);
		  const int sqrtVal = 64;
		  //const int sqrtVal = (int)(std::sqrt((float)hst_scene->state.iterations) + 0.5);
		  int y = iter / sqrtVal;
		  int x = iter % sqrtVal;
		  glm::vec2 sample = glm::vec2(u01(rng), u01(rng2));
		  //glm::vec2 sample = glm::vec2((x+ u01(rng)) / (sqrtVal*1.f), (y+ u01(rng2)) / (sqrtVal*1.f));
		  //glm::vec2 sample = glm::vec2((x+0.5 ) / (sqrtVal*1.f), (y+0.5 ) / (sqrtVal*1.f));
		  //std::cout << "itr" << iter << "sample x" << sample.x << "sample y" << sample.y << std::endl;
		  scatterRayStratified(pathSegments[idx], intersection.point, intersection.surfaceNormal, material, sample);
#else
		  scatterRay(pathSegments[idx], intersection.point, intersection.surfaceNormal, material, rng);
		  pathSegments[idx].remainingBounces--;
		  if (pathSegments[idx].remainingBounces == 0)
		  {
			  pathSegments[idx].color = glm::vec3(0.0f);
		  }
#if DIRECT_LIGHTING
		  //if (pathSegments[idx].remainingBounces == 0)
		  //{
			  //pathSegments[idx].color = glm::vec3(0.f, 0.f, 0.f);

			  thrust::uniform_int_distribution<float> u0l(0, num_lights);
			  int lightidx = u0l(rng);
			  Geom light = lights[lightidx];
			  Material materiallight = materials[light.materialid];
			  Ray ray;
			  ray.origin = intersection.point + intersection.surfaceNormal * EPSILON;
			  glm::vec3 pointonlight = sampleonlight(light, rng);
			  ray.direction = glm::normalize(pointonlight - ray.origin);
			  float t_min = FLT_MAX;
			  int hit_geom_index = -1;
			  rayIntersections(&ray, geoms, num_geoms, t_min, hit_geom_index);
			  /*if (hit_geom_index != -1)
			  {
				  if (geoms[hit_geom_index].materialid == light.materialid)
				  {
					  pathSegments[idx].color *= (materiallight.color * materiallight.emittance);
				  }
			  }*/
			  if (abs(t_min - glm::length(pointonlight - ray.origin)) < 1e-3f)
			  {
				  pathSegments[idx].color *= (material.color * materiallight.color * materiallight.emittance);
				  //pathSegments[idx].color = glm::vec3(0.f,0.f,1.f);
			  }
			  else
			  {
				  //pathSegments[idx].color = glm::vec3(0.f, 1.f, 0.f);
				  //pathSegments[idx].color == glm::vec3(0.0f);
			  }
			  pathSegments[idx].remainingBounces--;

		  //}
#endif
#endif
#if MSI
		  float pdf_direct = 0.f, pdf_scattering = 0.f, weight;
		  glm::vec3 wo, wi;
		  wo = -pathSegments[idx].ray.direction;
		  if (material.hasReflective || material.hasRefractive)
		  {
			  pathSegments[idx].specular = true;
		  }
		  //direct lighting
		  //choose a random light
		  thrust::uniform_int_distribution<float> u0l(0, num_lights);
		  int lightidx = u0l(rng);
		  Geom light = lights[lightidx];
		  Material materiallight = materials[light.materialid];
		  Ray ray;
		  ray.origin = intersection.point + intersection.surfaceNormal * EPSILON;

#endif
        //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        //pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
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
		if (!iterationPath.remainingBounces)//gather those whose remainingBounces == 0
		{
			image[iterationPath.pixelIndex] += iterationPath.color;
		}
	}
}

struct is_dead
{
	__host__ __device__
		bool operator()(const PathSegment& pathsegment)
	{
		return (pathsegment.remainingBounces == 0);
	}
};

struct compare_material
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
	{
		return (b.materialId > a.materialId);
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
	const int num_lights = hst_scene->lights.size();

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

#if MOTION_BLUR
	float step = 1 / (hst_scene->state.iterations*1.f);
	for (int i = 0; i < hst_scene->geoms.size(); i++)
	{
		if (hst_scene->geoms[i].motion)
		{
			hst_scene->geoms[i].translation += hst_scene->geoms[i].translate * step;
			hst_scene->geoms[i].rotation += hst_scene->geoms[i].rotate * step;
			hst_scene->geoms[i].transform = utilityCore::buildTransformationMatrix(hst_scene->geoms[i].translation, hst_scene->geoms[i].rotation, hst_scene->geoms[i].scale);
			hst_scene->geoms[i].inverseTransform = glm::inverse(hst_scene->geoms[i].transform);
			hst_scene->geoms[i].invTranspose = glm::inverseTranspose(hst_scene->geoms[i].transform);
		}
	}
	cudaMemcpy(dev_geoms, &(hst_scene->geoms)[0], hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

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

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if CACHE_FIRST_BOUNCE
	/*if (depth == 0 && cache_first_bounce)
	{
		cudaMemcpy(dev_intersections, dev_first_intersections, num_paths * sizeof(dev_first_intersections[0]),cudaMemcpyDeviceToDevice);
	}
	else
	{
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
	}*/
	if (depth == 0 && iter == 1)
	{
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		cudaMemcpy(dev_first_intersections, dev_intersections, num_paths * sizeof(dev_first_intersections[0]), cudaMemcpyDeviceToDevice);
	}
	else if (depth == 0 && iter != 1)
	{
		cudaMemcpy(dev_intersections, dev_first_intersections, num_paths * sizeof(dev_first_intersections[0]), cudaMemcpyDeviceToDevice);
	}
	else
	{
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
	}
#else
	computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		);
#endif

	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();
	depth++;

#if CACHE_FIRST_BOUNCE
	/*if (!cache_first_bounce)
	{
		cache_first_bounce = true;
		cudaMemcpy(dev_first_intersections, dev_intersections, num_paths * sizeof(dev_first_intersections[0]), cudaMemcpyDeviceToDevice);
	}*/
#endif

	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MATERIAL
	thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_material());
#endif

#if TIMER
	cudaEvent_t start, stop;
	if (iter == 1)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}
#endif

  shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
	num_lights,
	hst_scene->geoms.size(),
    dev_intersections,
    dev_paths,
    dev_materials,
	dev_lights,
	dev_geoms
  );

#if TIMER
  if (iter == 1)
  {
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float prev_elapsed_time_cpu_milliseconds = 0;
		cudaEventElapsedTime(&prev_elapsed_time_cpu_milliseconds, start, stop);
		//std::cout << "Elapsed time: " << prev_elapsed_time_cpu_milliseconds << "ms per iteration when depth = " << depth << std::endl;
		std::cout << prev_elapsed_time_cpu_milliseconds << std::endl;
  }
#endif

#if STREAM_COMPACTION
	dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);
	PathSegment* dev_path_end2 = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, is_dead());
	num_paths = dev_path_end2 - dev_paths;
	iterationComplete = (!num_paths|| depth>traceDepth); // TODO: should be based off stream compaction results.
#else
	iterationComplete = (depth > traceDepth); // TODO: should be based off stream compaction results.
#endif
	}

  // Assemble this iteration and apply it to the image
#if !STREAM_COMPACTION
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
#endif

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
