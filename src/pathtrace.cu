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

#include "device_launch_parameters.h"

#include "../stream_compaction/efficient.h"


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

static int *dev_indices = NULL;
static ShadeableIntersection *dev_intersections_cached = NULL;

// Sort by material
static int *dev_pathIndicesByMaterial = NULL;
static int *dev_intersectionIndicesByMaterial = NULL;

static Geom *dev_lights = NULL;



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

	// Compaction
	cudaMalloc(&dev_indices, pixelcount * sizeof(int));

	// Cached first bounce
	cudaMalloc(&dev_intersections_cached, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_cached, 0, pixelcount * sizeof(ShadeableIntersection));

	// Paths by material
	cudaMalloc(&dev_pathIndicesByMaterial, pixelcount * sizeof(int));
	cudaMemset(dev_pathIndicesByMaterial, 0, pixelcount * sizeof(int));
	// Intersections by material
	cudaMalloc(&dev_intersectionIndicesByMaterial, pixelcount * sizeof(int));
	cudaMemset(dev_intersectionIndicesByMaterial, 0, pixelcount * sizeof(int));

	// Get lights
	cudaMalloc(&dev_lights, scene->geoms.size() * sizeof(Geom));

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
	cudaFree(dev_intersections_cached);
	cudaFree(dev_pathIndicesByMaterial);
	cudaFree(dev_intersectionIndicesByMaterial);
	cudaFree(dev_lights);

	checkCUDAError("pathtraceFree");
}

#define PI 3.14159265358979323846f

__host__ __device__ 
glm::vec3 ConcentricSampleDisk(float sampleX, float sampleY)
{
	//float phi = 0.f;

	//float r = 0.f;
	//float u = 0.f;
	//float v = 0.f;

	//float a = 2.f * sampleX - 1.f;
	//float b = 2.f * sampleY - 1.f;

	//if (a > -b) {
	//	if (a > b) {
	//		r = a;
	//		phi = (PI / 4.f) * (b / a);
	//	}
	//	else {
	//		r = b;
	//		phi = (PI / 4.f) * (2.f - (a / b));
	//	}
	//}
	//else {
	//	if (a < b) {
	//		r = -a;
	//		phi = (PI / 4.f) * (4.f + (b / a));
	//	}
	//	else {
	//		r = -b;

	//		if (b != 0.f) {
	//			phi = (PI / 4.f) * (6.f - (a / b));
	//		}
	//		else {
	//			phi = 0.f;
	//		}
	//	}
	//}

	//u = r * cos(phi);
	//v = r * sin(phi);

	//return glm::vec3(u, v, 0.f);

	// http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
	float phi;
	float r;

	float a = 2.f * sampleX - 1.f;
	float b = 2.f * sampleY - 1.f;

	if (a * a > b * b) {
		r = a;
		phi = (PI / 4) * (b / a);
	}
	else {
		r = b;
		//phi = (PI / 4) * (a / b) + (PI / 2);
		phi = (PI / 2) - (PI / 4) * (a / b);
	}

	return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), 0.f);
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
#ifndef AA	
#define AA
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		glm::vec2 jitter(u01(rng), u01(rng));

		float _x = (float)x + jitter.x;
		float _y = (float)y + jitter.y;
#else 
		float _x = x;
		float _y = y;
#endif

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)_y - (float)cam.resolution.y * 0.5f)
		);

		// Depth of Field
		if (cam.lensRadius > 0.f) {
			// Sample point on lens
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);
			glm::vec3 pLens = cam.lensRadius * ConcentricSampleDisk(u01(rng), u01(rng)); 

			// Compute point on plane of focus
			glm::vec3 pFocus = cam.focalDistance * segment.ray.direction + segment.ray.origin;
			glm::vec3 aperaturePoint = segment.ray.origin + (cam.up * pLens.y) + (cam.right + pLens.x);

			// Update ray
			segment.ray.origin = aperaturePoint;
			segment.ray.direction = glm::normalize(pFocus - aperaturePoint);
		}

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
			else if (geom.type == SQUARE)
			{
				t = squareIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].intersectPoint = intersect_point;
		}
	}
}

__host__ __device__ void Intersect(
	PathSegment &pathSegment,
	Geom *geoms,
	int geoms_size,
	ShadeableIntersection &intersection)
{
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
		intersection.t = -1.0f;
	}
	else
	{
		//The ray hits something
		intersection.t = t_min;
		intersection.materialId = geoms[hit_geom_index].materialid;
		intersection.surfaceNormal = normal;
		intersection.intersectPoint = intersect_point;
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
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
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

// Shading kernel with BSDF evaluations for:
//		- Ideal diffuse surfaces (cosine-weighted scattering)
//		- Perfectly specular-reflective surfaces
__global__ void shadeMaterialNaive(
	int iter,
	int num_paths,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	Material * materials,
	int depth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}

		// Check if the ray hits anything
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t <= 0.0f) {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}

		// Use this without stream compaction
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
		//thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		// See what kind of object the ray hit.
		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// Ray hits the light.
		if (material.emittance > 0.0f) {
			pathSegments[idx].color *= (materialColor * material.emittance);
			pathSegments[idx].remainingBounces = 0;
		}
		// Ray hits an object.
		// Scatter the ray to get a new ray and update the color
		else {
			scatterRay(pathSegments[idx], intersection.intersectPoint, intersection.surfaceNormal, material, rng);
			pathSegments[idx].remainingBounces--;
		}
	}
}

__global__ void shadeMaterialDirect(
	int iter,
	int num_paths,
	ShadeableIntersection *shadeableIntersections,
	PathSegment *pathSegments,
	Material *materials,
	Geom *geoms,
	int num_geoms,
	int depth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}

		// Check if the ray hits anything
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t <= 0.0f) {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}

		// Use this without stream compaction
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
		//thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		// See what kind of object the ray hit.
		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// Ray hits the light.
		if (material.emittance > 0.0f) {
			pathSegments[idx].color *= (materialColor * material.emittance);
			pathSegments[idx].remainingBounces = 0;
		}
		// Ray hits an object.
		// Scatter the ray to get a new ray and update the color
		else {
			float pdf;
			glm::vec3 wi;

			// Pick a random light in the scene
			// Cornell has 1 light at geoms[0]
			int numLights = 1;
			Geom light = geoms[0];

			// Sample light
			// Might have to convert point 
			//glm::vec3 li = Sample_Li(intersection, light, wi, pdf, rng);

			// Sample a point on the light
			glm::vec3 samplePoint = SphereSample(rng);
			// Get the light's position based on the sample
			glm::vec3 lightPos(light.transform * glm::vec4(samplePoint, 1.f));

			// Sample the object
			glm::vec3 f = bsdf_f(pathSegments[idx], material);

			// Test for shadows
			PathSegment shadowFeeler;
			shadowFeeler.ray.direction = glm::normalize(lightPos - intersection.intersectPoint);
			shadowFeeler.ray.origin = intersection.intersectPoint + (EPSILON * shadowFeeler.ray.direction);

			ShadeableIntersection isx;
			Intersect(shadowFeeler, geoms, num_geoms, isx);
			
			float emit = 0.f;
			// Occluded by object
			if (isx.t > 0.f && materials[isx.materialId].emittance <= 0.f) {
				pathSegments[idx].color = glm::vec3(0.f);
			}
			else {
				if (glm::dot(lightPos, -shadowFeeler.ray.direction) > 0.f) {
					emit = materials[light.materialid].emittance;
				} 
				
				pathSegments[idx].color *= f * emit * AbsDot(intersection.surfaceNormal, glm::normalize(shadowFeeler.ray.direction));
			}

			// Direct lighting only does one bounce.
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void scanPaths(int num_paths, PathSegment *pathSegments, int *dev_indices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		if (pathSegments[idx].remainingBounces > 0) {
			dev_indices[idx] = 1;
		}
		else if (pathSegments[idx].remainingBounces == 0) {
			dev_indices[idx] = 0;
		}
	}
}

__global__ void sortByMaterial(int num_paths, ShadeableIntersection *intersections, int *pathIndices, int *intersectionIndices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		int materialId = intersections[idx].materialId;
		pathIndices[idx] = materialId;
		intersectionIndices[idx] = materialId;
	}
}

struct endPath
{
	__host__ __device__ 
	bool operator()(const PathSegment &pathSegment)
	{
		return pathSegment.remainingBounces > 0;
	}
};

struct isLight
{
	__host__ __device__
		bool operator()(const Geom &geometry)
	{
		//return dev_materials[geometry.materialid].emittance > 0;
		return false;
	}
};

//__global__ void getLights(Geom *geoms, Material *materials, Geom *lights)
//{
//
//}

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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	int numActivePaths = num_paths;

	// TODO
	// Get lights from scene
	//thrust::copy_if(thrust::device, dev_geoms, dev_geoms + hst_scene->geoms.size(), dev_lights, isLight());
	//Geom *geoms = thrust::remove_if(thrust::device, dev_geoms, dev_geoms + 4, isLight());
	//Geom *lights = new Geom[hst_scene->geoms.size()];

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	//while (depth < traceDepth || numActivePaths > 0) {
	while (!iterationComplete) {
		
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		//printf("TRACE DEPTH: %d\nITERATION: %d\nDEPTH: %d\n-------\n", traceDepth, iter, depth);

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#define CACHE false
#define SORTBYMATERIAL false

		// Store the very first bounce into dev_intersections_cached.
		if (CACHE && depth == 0 && iter == 1) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections_cached
				);
			checkCUDAError("trace one bounce");
		}
		// Get new intersections from the new scattered rays.
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
		}

		cudaDeviceSynchronize();

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		// At each iteration on the first bounce, use the cached intersection.
		if (CACHE && depth == 0 && iter == 1) {
			// Sort paths by material
			if (SORTBYMATERIAL) {
				sortByMaterial << < numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections_cached, dev_pathIndicesByMaterial, dev_intersectionIndicesByMaterial);

				thrust::sort_by_key(thrust::device, dev_pathIndicesByMaterial, dev_pathIndicesByMaterial + num_paths, dev_paths);
				thrust::sort_by_key(thrust::device, dev_intersectionIndicesByMaterial, dev_intersectionIndicesByMaterial + num_paths, dev_intersections_cached);
			}

			//shadeMaterialDirect << < numblocksPathSegmentTracing, blockSize1d >> > (iter, num_paths, dev_intersections_cached, dev_paths, dev_materials, dev_geoms, hst_scene->geoms.size(), depth);
			shadeMaterialNaive << < numblocksPathSegmentTracing, blockSize1d >> > (iter, num_paths, dev_intersections_cached, dev_paths, dev_materials, depth);
		}
		// Otherwise use the new ray.
		else {
			// Sort paths by material
			if (SORTBYMATERIAL) {
				sortByMaterial << < numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_pathIndicesByMaterial, dev_intersectionIndicesByMaterial);

				thrust::sort_by_key(thrust::device, dev_pathIndicesByMaterial, dev_pathIndicesByMaterial + num_paths, dev_paths);
				thrust::sort_by_key(thrust::device, dev_intersectionIndicesByMaterial, dev_intersectionIndicesByMaterial + num_paths, dev_intersections);
			}

			//shadeMaterialDirect << < numblocksPathSegmentTracing, blockSize1d >> > (iter, num_paths, dev_intersections, dev_paths, dev_materials, dev_geoms, hst_scene->geoms.size(), depth);
			shadeMaterialNaive << < numblocksPathSegmentTracing, blockSize1d >> > (iter, num_paths, dev_intersections, dev_paths, dev_materials, depth);
		}

		// Compact paths
		// Puts all the paths with bounces > 0 towards the front of the array
		//PathSegment *activePaths = thrust::partition(thrust::device, dev_paths, dev_paths + numActivePaths, endPath());
		//// Update the number of active paths so we can only operate on those the next loop.
		//numActivePaths = activePaths - dev_paths;
		
		depth++;
		iterationComplete = (depth >= traceDepth || numActivePaths == 0);
	}

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
}
