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

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
//#define DIRECT_LIGHTING
#define MATERIAL_SORT

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

struct not_zero {
	__host__ __device__ bool operator()(const PathSegment& path) {
		return path.remainingBounces > 0;
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
        color.x = glm::clamp((int) (pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z * 255.0), 0, 255);

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
static int * dev_materialType = NULL;
// TODO: static variables for device memory, any extra info you need, etc
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

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_materialType, pixelcount * sizeof(int));
	cudaMemset(dev_materialType, -1, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
	cudaFree(dev_lights);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	cudaFree(dev_materialType);

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

		// Antialiasing by jittering
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		float jitterX = u01(rng);
		float jitterY = u01(rng);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((x + jitterX) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((y + jitterY) - (float)cam.resolution.y * 0.5f)
			);

		// Depth of field
		float lensRadius = 0.0f;
		float focalDistance = 10.0f;
		if (lensRadius > 0.0f) {
			float u1 = u01(rng);
			float u2 = u01(rng);
			float r = sqrt(u1);
			float theta = TWO_PI * u2;
			glm::vec2 sampleDisk(r * glm::cos(theta), r * glm::sin(theta));
			float lensU = lensRadius * sampleDisk.x;
			float lensV = lensRadius * sampleDisk.y;
			glm::vec3 focalPoint = segment.ray.origin + (segment.ray.direction * focalDistance);

			segment.ray.origin += (cam.right * lensU) + (cam.up * lensV);
			segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__device__ float computeRayIntersection(Ray ray, Geom *geoms, int geoms_size) {
	float t;
	float t_min = FLT_MAX;
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

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
		}
	}

	if (hit_geom_index == -1)
	{
		return -1.0f;
	}
	else
	{
		//The ray hits something
		return t_min;
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
	, int * materialType
	, Material * materials
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

		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;

			materialType[path_index] = NONE;
		}
		else {
			//The ray hits something
			int materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = materialId;
			intersections[path_index].surfaceNormal = normal;

			Material material = materials[materialId];
			if (material.hasReflective) {
				materialType[path_index] = REFLECTIVE;
			}
			else {
				materialType[path_index] = DIFFUSE;
			}
			if (material.emittance > 0.0f) {
				materialType[path_index] = LIGHT;
			}
		}
	}
}

__global__ void shadeAllMaterials(int iter, int depth, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Material* materials, Geom* lights, int num_lights, Geom* geoms, int num_geoms) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < num_paths) {
		if (!STREAM_COMPACTION && pathSegments[index].remainingBounces == 0) {
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[index];

		if (intersection.t > 0) {
			// Intersected with an object
			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			if (material.emittance > 0.0f) {
				// Hit a light source
				pathSegments[index].color *= (materialColor * material.emittance);
				pathSegments[index].remainingBounces = 0;
			}
			else {
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);
				glm::vec3 intersectionPoint = pathSegments[index].ray.origin + (pathSegments[index].ray.direction * intersection.t);

				glm::vec3 brdf;
				glm::vec3 wi;
				float absdot;
				float pdf;
				if (material.hasReflective) {
					// Reflective
					brdf = material.color;
					wi = glm::reflect(pathSegments[index].ray.direction, intersection.surfaceNormal);
					absdot = 1.0f;
					pdf = 1.0f;
				}
				else {
					// Diffuse
					brdf = materialColor / PI;
					wi = cosineWeightedSample(intersection.surfaceNormal, rng);
					float wDotN = glm::dot(wi, intersection.surfaceNormal);
					absdot = glm::abs(wDotN);
					pdf = wDotN / PI;
				}
				glm::vec3 brdfSample = (brdf * absdot) / pdf;

				// Update pathSegment state
				pathSegments[index].color *= brdfSample;
				pathSegments[index].ray.origin = intersectionPoint + (intersection.surfaceNormal * EPSILON);
				pathSegments[index].ray.direction = wi;
				pathSegments[index].remainingBounces--;
				if (pathSegments[index].remainingBounces == 0) {
#ifdef DIRECT_LIGHTING
					// Direct lighting
					thrust::uniform_real_distribution<float> u01(0, 1);
					int lightIndex = glm::floor(num_lights * u01(rng));

					Geom light = lights[lightIndex];
					Material lightMaterial = materials[light.materialid];
					Ray ray;
					glm::vec3 dummyPoint(0.0f);
					glm::vec3 dummyNormal(0.0f);
					bool dummyBool = false;
					ray.origin = intersectionPoint + (intersection.surfaceNormal * EPSILON);
					glm::vec3 sampleRay;
					if (light.type == SPHERE) {
						glm::vec3 lightSample = generateSphereSample(light, rng);
						sampleRay = lightSample - ray.origin;
						ray.direction = glm::normalize(sampleRay);
					}
					else if (light.type == CUBE) {
						glm::vec3 lightSample = generateCubeSample(light, rng);
						sampleRay = lightSample - ray.origin;
						ray.direction = glm::normalize(sampleRay);
					}

					float t = computeRayIntersection(ray, geoms, num_geoms);

					float length = glm::length(sampleRay);
					if ((t > length - 0.1f) && (t < length + 0.1f)) {
						// Light is visible
						pathSegments[index].color *= lightMaterial.color * lightMaterial.emittance;
					}
					else {
						// Light is blocked
						pathSegments[index].color = glm::vec3(0.0f);
					}
#else
					pathSegments[index].color = glm::vec3(0.0f);
#endif
				}
			}
		}
		else {
			// No intersection
			pathSegments[index].color = glm::vec3(0.0f);
			pathSegments[index].remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths, int iter)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (iter > 0) {
			glm::vec3 oldColor = image[iterationPath.pixelIndex] * (float)(iter);
			image[iterationPath.pixelIndex] = (iterationPath.color + oldColor) / (float)(iter + 1);
		}
		else {
			image[iterationPath.pixelIndex] = iterationPath.color;
		}
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
	const int lightcount = hst_scene->lights.size();

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

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_materialType
			, dev_materials
			);
		checkCUDAError("Failed to trace one bounce");
		cudaDeviceSynchronize();
		depth++;

#ifdef MATERIAL_SORT
		// TODO: Sort path segments by material
#endif


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		shadeAllMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			depth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_lights,
			lightcount,
			dev_geoms,
			hst_scene->geoms.size()
		);

		if (STREAM_COMPACTION) {
			PathSegment* compacted = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, not_zero());
			num_paths = compacted - dev_paths;
			iterationComplete = (num_paths == 0) || (depth == traceDepth);
		}
		else {
			iterationComplete = (depth == traceDepth);
		}
	}

	num_paths = dev_path_end - dev_paths;

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths, iter);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
