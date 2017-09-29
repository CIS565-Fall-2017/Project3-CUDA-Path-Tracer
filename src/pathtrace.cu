#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define ANIMATION 0
#define MOTIONBLUR 0
#define BOUNDING_VOLUME_CULLING 0
#define STREAMCOMPACTION 1
#define AA 0
#define DOF 0
#define SORTMATERIALS 0
#define CACHE 0

#define COMPACT 0
#define NOT_COMPACT 1000

#define LIGHT 1

typedef int stream_t;
typedef thrust::tuple<PathSegment, ShadeableIntersection> ps_t;
typedef thrust::zip_iterator <thrust::tuple<thrust::device_ptr<PathSegment>,
	thrust::device_ptr<ShadeableIntersection>>> iter;

#define deg2rag(deg) (deg * PI / 180.0)

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

static int ntris = 0;
static Tri * dev_tris = NULL;
static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_intersections_cache = NULL;

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

	cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));

	Tri *tris_arr = hst_scene->tris.data();
	ntris = hst_scene->tris.size();

	cudaMalloc(&dev_tris, ntris * sizeof(Tri));
	cudaMemcpy(dev_tris, tris_arr, ntris * sizeof(Tri), cudaMemcpyHostToDevice);


    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    
	cudaFree(dev_intersections_cache);
	cudaFree(dev_tris);
    checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec2 ConcentricSample(const float u, const float v)
{
	float r;
	float theta;
	float sx = 2 * u - 1;
	float sy = 2 * v - 1;
	if (sx == 0 && sy == 0) return glm::vec2(0);
	if (sx > -sy) {
		if (sx > sy) {
			r = sx;
			theta = (sy > 0) ? sy / r : 8.0f + sy / r;
		} else {
			r = sy;
			theta = 2.0f - sx / r;
		}
	} else {
		if (sx <= sy) {
			r = -sx;
			theta = 4.0f - sy / r;
		} else {
			r = -sy;
			theta = 6.0f + sx / r;
		}
	}

	theta *= PI / 4.0f;
	float a = cosf(theta) * r;
	float b = sinf(theta) * r;
	return glm::vec2(a, b);
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

	float _x = x;
	float _y = y;
	// TODO: implement antialiasing by jittering the ray
	#if AA
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		_x = (float)x + u01(rng);
		_y = (float)y + u01(rng);
	#endif

	segment.ray.direction = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * (_x - (float)cam.resolution.x * 0.5f)
		- cam.up * cam.pixelLength.y * (_y - (float)cam.resolution.y * 0.5f)
	);


	#if DOF
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		glm::vec2 sm = ConcentricSample(u01(rng), u01(rng)) * cam.ap;
		float f = cam.f / abs(segment.ray.direction[2]);
		glm::vec3 _f = segment.ray.direction * f + segment.ray.origin;
		segment.ray.origin += glm::vec3(sm.x, sm.y, 0.f);
		segment.ray.direction = glm::normalize(_f - segment.ray.origin);
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
	, int ntris
	, Tri * tris
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{

		PathSegment pathSegment = pathSegments[path_index];

		if (pathSegment.remainingBounces <= 0) return;

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		#pragma unroll
		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = i;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = i;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}
			}
			else if (geom.type == TRIS) { 
				#if BOUNDING_VOLUME_CULLING
					if (boxBoundingVolumeTest(geom, geom.a, geom.b, pathSegment.ray)) {

						#pragma unroll
						for (int j = 0; j < ntris; j++) {
							Tri ti = tris[j];
							t = triIntersectionTest(ti, geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

							if (t > 0.0f && t_min > t)
							{
								t_min = t;
								hit_geom_index = i;
								intersect_point = tmp_intersect;
								normal = tmp_normal;
							}
						}
					}
				#else
					for (int j = 0; j < ntris; j++) {
						Tri ti = tris[j];
						t = triIntersectionTest(ti, geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

						if (t > 0.0f && t_min > t)
						{
							t_min = t;
							hit_geom_index = i;
							intersect_point = tmp_intersect;
							normal = tmp_normal;
						}
					}
				#endif		
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
			intersections[path_index].outside = outside;
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
  int iter,
	int depth
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
  {
	PathSegment &pseg = pathSegments[idx];
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

	  if (pseg.remainingBounces > 0) {
		  glm::vec3 intersect = pseg.ray.origin + intersection.t * pseg.ray.direction;
		  scatterRay(pseg, intersect, intersection.surfaceNormal, intersection.outside, material, rng);
	  }

    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
		pseg.color = glm::vec3(0.0f);
		pseg.remainingBounces = 0;
    }
  }
}


struct term_s
{
	__host__ __device__
	bool operator()(const PathSegment & pathSeg)
	{
		return (pathSeg.remainingBounces <= 0);
	}
};

struct mat_s
{
	__host__ __device__
	bool operator()(const ps_t & a, const ps_t & b)
	{
		return a.get<1>().materialId < b.get<1>().materialId;
	}
};

void SortMat(int num_path, PathSegment *dev_path, ShadeableIntersection *dev_intersections)
{
	using namespace thrust;
	device_ptr<PathSegment> thrust_dev_pseg(dev_path);
	device_ptr<ShadeableIntersection> thrust_dev_itsct(dev_intersections);
	iter s = make_zip_iterator(thrust::make_tuple(thrust_dev_pseg, thrust_dev_itsct));
	iter e = s + num_path;
	thrust::sort(s, e, mat_s());
}

// Add the current iteration's output to the overall image
template<int m>
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths && iterationPaths[index].remainingBounces <= m)
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

	#if MOTIONBLUR
	Geom *gs_copy;
	Geom *gs = hst_scene->geoms.data();

	gs_copy = (Geom *)malloc(hst_scene->geoms.size() * sizeof(Geom));
	memcpy(gs_copy, gs, hst_scene->geoms.size() * sizeof(Geom));

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, frame, traceDepth);
	thrust::uniform_real_distribution<float> u01(0, 1);
	for (size_t i = 0; i < hst_scene->geoms.size(); i++)
	{
		float aTime = hst_scene->aniTime;
		Geom &g = gs_copy[i];
		float p = u01(rng) * 0.2 - 0.1;
		float t_p = max(0.f, min(1.f, p + aTime));
		g.transform = glm::translate(g.transform, g.mt * t_p);
		g.transform = glm::rotate(g.transform, (float)deg2rag(g.mr.x) * t_p, glm::vec3(1,0,0));
		g.transform = glm::rotate(g.transform, (float)deg2rag(g.mr.x) * t_p, glm::vec3(0,1,0));
		g.transform = glm::rotate(g.transform, (float)deg2rag(g.mr.x) * t_p, glm::vec3(0,0,1));

		g.inverseTransform = glm::inverse(g.transform);
		g.invTranspose = glm::inverse(glm::transpose(g.transform));
	}
	cudaMemcpy(dev_geoms, gs_copy, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	#else
		#if ANIMATION
			Geom *gs_copy;
			Geom *gs = hst_scene->geoms.data();

			gs_copy = (Geom *)malloc(hst_scene->geoms.size() * sizeof(Geom));
			memcpy(gs_copy, gs, hst_scene->geoms.size() * sizeof(Geom));

			for (size_t i = 0; i < hst_scene->geoms.size(); i++)
			{
				float aTime = hst_scene->aniTime;
				Geom &g = gs_copy[i];
				float t_p = max(0.f, min(1.f, aTime));
				g.transform = glm::translate(g.transform, g.mt * t_p);
				g.transform = glm::rotate(g.transform, (float)deg2rag(g.mr.x) * t_p, glm::vec3(1, 0, 0));
				g.transform = glm::rotate(g.transform, (float)deg2rag(g.mr.y) * t_p, glm::vec3(0, 1, 0));
				g.transform = glm::rotate(g.transform, (float)deg2rag(g.mr.z) * t_p, glm::vec3(0, 0, 1));

				g.inverseTransform = glm::inverse(g.transform);
				g.invTranspose = glm::inverse(glm::transpose(g.transform));
			}
			cudaMemcpy(dev_geoms, gs_copy, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
		#endif // ANIMATION
	#endif
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
	int rest_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// tracing
	dim3 numblocksPathSegmentTracing = (rest_paths + blockSize1d - 1) / blockSize1d;
	

	#if !CACHE
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, rest_paths
			, dev_paths
			, dev_geoms
			, ntris
			, dev_tris
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;
	#else
		if ((iter == 1 && depth == 0) || (depth > 0)) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, rest_paths
				, dev_paths
				, dev_geoms
				, ntris
				, dev_tris
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}

		if (iter == 1 && depth == 0) {
			cudaMemcpy(dev_intersections_cache, dev_intersections, rest_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}

		if (iter > 1 && depth == 0) {
			cudaMemcpy(dev_intersections, dev_intersections_cache, rest_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}

		depth++;
	#endif


	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.
#if SORTMATERIALS
	SortMat(rest_paths, dev_paths, dev_intersections);
#endif // SORTMATERIALS

  shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
	depth,
	  rest_paths,
    dev_intersections,
    dev_paths,
    dev_materials
  );
  
#if STREAMCOMPACTION
	finalGather<COMPACT><<<numblocksPathSegmentTracing, blockSize1d >>> (rest_paths, dev_image, dev_paths);
	cudaDeviceSynchronize();
	PathSegment *np = thrust::remove_if(thrust::device, dev_paths, dev_paths + rest_paths, term_s());
	if (np != NULL) rest_paths = np - dev_paths;
	else rest_paths = 0;

	//printf("Rest_Path: %i\n", rest_paths);
	
#endif // STEAMCOMPACTION

  iterationComplete = (rest_paths <= 0 || depth > traceDepth);

	}

#if !STREAMCOMPACTION
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<NOT_COMPACT> << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);
#endif // STREAMCOMPACTION

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
