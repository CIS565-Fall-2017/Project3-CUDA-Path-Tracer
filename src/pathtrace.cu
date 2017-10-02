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
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "timer.h"

#include "../stream_compaction/common.h"
#include "bvh.h"



#define ERRORCHECK 1

#define COMPACT 1
#define MATSORT 0
#define CACHE_CAMERA_RAYS 0
#define AA_JITTER 1
#define BVH_DEPTH 16 /*
		0
	  /   \
	1		1

*/
#define NUMBUCKETS 4
#define BVH 1
#define BVHDEBUG 0
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

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
static PathSegment * dev_cached_camera_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static BVHNode * dev_BVHnodes = NULL;
static int * dev_geomBVHIndices = NULL;
static int * dev_BVHstart = NULL;
static int * dev_BVHend = NULL;
static int * dev_matIndicesI = NULL;
static int * dev_matIndicesP = NULL;

thrust::device_ptr<int> dev_thrust_matIndicesI;
thrust::device_ptr<int> dev_thrust_matIndicesP;
thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections;
thrust::device_ptr<PathSegment> dev_thrust_paths;
// TODO: static variables for device memory, any extra info you need, etc
// ...


void createGeomBounds() {
	int num_geoms = hst_scene->geoms.size();
	const int blockSize1d = 128;
	dim3 numblocksGeoms = (num_geoms + blockSize1d - 1) / blockSize1d;
	kernCalculateBounds << < numblocksGeoms, blockSize1d >> > (num_geoms, dev_geoms);
	//Geom * testarr = new Geom[num_geoms];
	//cudaMemcpy(testarr, dev_geoms, num_geoms, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < num_geoms; i++) {
	//	Geom test = testarr[i];
	//	glm::vec3 maxb = test.maxb;
	//}
}

void constructBVHTree() {
	int num_BVHnodes = (1 << (BVH_DEPTH + 1)) - 1;
	int num_geoms = hst_scene->geoms.size();
	/*int * BVHstart = new int[num_BVHnodes];
	int * BVHend = new int[num_BVHnodes];*/
	std::vector<int> BVHstart;
	std::vector<int> BVHend;
	BVHstart.resize(num_BVHnodes);
	BVHend.resize(num_BVHnodes);
	BVHstart[0] = 0;
	BVHend[0] = num_geoms - 1;
	//BVHNode * BVHnodes = new BVHNode[num_BVHnodes];
	std::vector<BVHNode> BVHnodes;
	BVHnodes.resize(num_BVHnodes);
	//printf("test: %i %i", BVHstart[0], BVHend[0]);
	glm::vec3 maxb = thrust::transform_reduce(thrust::device, dev_geoms, dev_geoms + num_geoms, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
	glm::vec3 minb = thrust::transform_reduce(thrust::device, dev_geoms, dev_geoms + num_geoms, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());
	BVHnodes[0] = BVHNode();
	BVHnodes[0].maxb = maxb;
	BVHnodes[0].minb = minb;
	int curr_axis = 0;
	for (int depth = 0; depth <= BVH_DEPTH - 1; depth++) { //depth <= BVH_DEPTH
		int num_nodes_at_depth = 1 << depth;
		int offset = (1 << depth) - 1;
		for (int n = 0; n < num_nodes_at_depth; n++) { // n < num_nodes_at_depth
			int curr_bvh = offset + n;
			int axis = curr_axis;
			float val;
			float repeat = true;
			int start_idx = BVHstart[curr_bvh];
			int end_idx = BVHend[curr_bvh];
			if (start_idx == end_idx) {
				continue;
			}
			if (repeat) {
				int buck_off = 0;
				float buck_width = (BVHnodes[curr_bvh].maxb[axis] - BVHnodes[curr_bvh].minb[axis])/ (float)NUMBUCKETS;
				val = BVHnodes[curr_bvh].minb[axis];
				float buckCost[NUMBUCKETS];
				float buckIdx[NUMBUCKETS];
				float total_sa = thrust::transform_reduce(thrust::device, dev_geoms + start_idx, dev_geoms + end_idx + 1, get_sa(), 0.0f, sum_sa());
				for (int buck = 0; buck < NUMBUCKETS; buck++) {
					val += buck_width;
					Geom* end = thrust::partition(thrust::device, dev_geoms + start_idx + buck_off, dev_geoms + end_idx + 1, less_than_axis(axis, val));
					int num_left = end - (dev_geoms + start_idx + buck_off);
					float left_sa = thrust::transform_reduce(thrust::device, dev_geoms + start_idx + buck_off, dev_geoms + start_idx + buck_off + num_left, get_sa(), 0.0f, sum_sa());
					buck_off += num_left;
					//printf("left_sa %f\n", left_sa);
					buckIdx[buck] = buck_off;
					buckCost[buck] = left_sa;
				}
				float total_cost = 0;
				float min_cost = INFINITY;
				int min_idx = -1;
				for (int buck = 0; buck < NUMBUCKETS; buck++) {
					total_cost += buckCost[buck];
					float cost = fabsf((total_sa / 2.0f) - total_cost);
					if (cost < min_cost) {
						min_cost = cost;
						min_idx = buckIdx[buck];
					}
				}
				int next_offset = (1 << (depth + 1)) - 1;
				int l_idx = next_offset + (2 * n);
				BVHstart[l_idx] = start_idx;
				BVHend[l_idx] = start_idx + min_idx - 1;
				int r_idx = next_offset + (2 * n) + 1;
				BVHstart[r_idx] = start_idx + min_idx;
				BVHend[r_idx] = end_idx;
				glm::vec3 lmaxb = thrust::transform_reduce(thrust::device, dev_geoms + start_idx, dev_geoms + BVHend[l_idx] + 1, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
				glm::vec3 lminb = thrust::transform_reduce(thrust::device, dev_geoms + start_idx, dev_geoms + BVHend[l_idx] + 1, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());

				glm::vec3 rmaxb = thrust::transform_reduce(thrust::device, dev_geoms + BVHstart[r_idx], dev_geoms + end_idx + 1, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
				glm::vec3 rminb = thrust::transform_reduce(thrust::device, dev_geoms + BVHstart[r_idx], dev_geoms + end_idx + 1, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());
				BVHnodes[l_idx] = BVHNode();
				BVHnodes[r_idx] = BVHNode();
				BVHnodes[l_idx].id = l_idx;
				BVHnodes[r_idx].id = r_idx;
				BVHnodes[l_idx].start = BVHstart[l_idx];
				BVHnodes[l_idx].end = BVHend[l_idx];
				BVHnodes[r_idx].start = BVHstart[r_idx];
				BVHnodes[r_idx].end = BVHend[r_idx];
				BVHnodes[l_idx].maxb = lmaxb;
				BVHnodes[l_idx].minb = lminb;
				BVHnodes[r_idx].maxb = rmaxb;
				BVHnodes[r_idx].minb = rminb;
				BVHnodes[curr_bvh].is_leaf = false;
				BVHnodes[curr_bvh].child1id = l_idx;
				BVHnodes[curr_bvh].child2id = r_idx;

				repeat = false;
			}

		}
		curr_axis = (curr_axis + 1) % 3;
	}

	const int blockSize1d1 = 128;//num_geoms
	dim3 numblocksGeoms = (num_geoms + blockSize1d1 - 1) / blockSize1d1;
	printf("after BVH construct \n");
	kernCheckBounds << < numblocksGeoms, blockSize1d1 >> > (num_geoms, dev_geoms);

	const int blockSize1d = 128; //num_BVHnodes
	cudaMemcpy(dev_BVHnodes, BVHnodes.data(), num_BVHnodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_BVHstart, BVHstart.data(), num_BVHnodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_BVHend, BVHend.data(), num_BVHnodes * sizeof(int), cudaMemcpyHostToDevice);
	dim3 numblocksBVH = (num_BVHnodes + blockSize1d - 1) / blockSize1d;
	kernSetBVHTransform <<< numblocksBVH, blockSize1d >>> (num_BVHnodes, dev_BVHnodes);

}

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_cached_camera_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if MATSORT
	cudaMalloc(&dev_matIndicesI, pixelcount * sizeof(int));
	cudaMalloc(&dev_matIndicesP, pixelcount * sizeof(int));
#endif

#if BVH
		int num_BVHnodes = (1 << (BVH_DEPTH + 1)) - 1;
		int num_BVHleafnodes = 1 << (BVH_DEPTH - 1);

		cudaMalloc(&dev_BVHnodes, num_BVHnodes * sizeof(BVHNode));
		cudaMalloc(&dev_BVHstart, num_BVHnodes * sizeof(int));
		cudaMalloc(&dev_BVHend, num_BVHnodes * sizeof(int));
		cudaMalloc(&dev_geomBVHIndices, scene->geoms.size() * sizeof(int));
		// TODO: initialize any extra device memeory you need

		printf("BVH depth: %i", BVH_DEPTH);
		startCpuTimer();
		createGeomBounds();
		constructBVHTree();
		endCpuTimer();
		printf("BVH construction time:");
		printTime();
#endif
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_cached_camera_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_BVHnodes);
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

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray

		float aa_offx = 0.0;
		float aa_offy = 0.0;
		if (AA_JITTER && !CACHE_CAMERA_RAYS) { //cannot cache rays and jitter anti alias
			thrust::default_random_engine rng_j = makeSeededRandomEngine(iter, index, 0);
			thrust::uniform_real_distribution<float> u_j(-0.5f, 0.5f);
			aa_offx = u_j(rng_j);
			aa_offy = u_j(rng_j);
		}

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + aa_offx)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + aa_offy)
		);
		if (cam.dof) {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, iter*index, 0);
			thrust::uniform_real_distribution<float> u01(-1, 1);
			thrust::uniform_real_distribution<float> u02(-1, 1);
			float ap = cam.lensrad;
			float focal_dist = cam.focal_dist;
			glm::vec2 off = calculateRandomUniformPointDisc(ap, u01(rng), u02(rng));
			glm::vec3 off_pos = segment.ray.origin + (cam.up*off.y) + (cam.right*off.x);
			glm::vec3 focal_point = segment.ray.origin + (focal_dist * segment.ray.direction);
			segment.ray.direction = glm::normalize(focal_point - off_pos);
			segment.ray.origin = off_pos;
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

		if (pathSegment.pixelIndex == 400) {
			int pixelIndex = pathSegment.pixelIndex;
			pixelIndex++;
		}

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
			else if (geom.type == TRIANGLE) {
				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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

__global__ void shadeDebugMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
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
			pathSegments[idx].color = intersection.surfaceNormal;
		}
		else {
			pathSegments[idx].color = intersection.surfaceNormal;
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void shadeAnyMaterial(
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

		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			int pixelIdx = pathSegments[idx].pixelIndex;
			int remaining = pathSegments[idx].remainingBounces;
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx * remaining, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

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
				float t = intersection.t;
				glm::vec3 norm = intersection.surfaceNormal;
				glm::vec3 new_dir;
				Ray old_ray = pathSegments[idx].ray;
				if (material.hasReflective) {
					new_dir = glm::reflect(old_ray.direction, norm);
				}
				else {
					float test = u01(rng);
					new_dir = glm::normalize(calculateRandomDirectionInHemisphere(norm, rng));
					pathSegments[idx].color *= /*fabsf(glm::dot(norm,new_dir)) **/ materialColor;
				}
				glm::vec3 old_inter = old_ray.origin + (old_ray.direction * t);
				Ray new_ray = { old_inter + norm * 0.0001f, new_dir };
				pathSegments[idx].ray.direction = new_dir;
				pathSegments[idx].ray.origin = old_inter + norm * 0.00001f;
				/*if (pixelIdx == 160300) {
					printf("iter %i \n", iter);
					printf("ypos %f %f %f \n", old_inter.x, old_inter.y, old_inter.z);
					printf("ynorm %f %f %f \n", norm.x, norm.y, norm.z);
					printf("ydir %f %f %f \n", new_dir.x, new_dir.y, new_dir.z);
				}*/
				if (--pathSegments[idx].remainingBounces <= 0) {
					pathSegments[idx].color = glm::vec3(0.0f);
				}
				
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

__global__ void kernSetMatIDArr(int n, ShadeableIntersection *intersections, int *matIndicesI, int *matIndicesP) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n)
	{
		matIndicesI[index] = intersections[index].materialId;
		matIndicesP[index] = intersections[index].materialId;
	}
}

struct has_bounces {
	__host__ __device__ bool operator() (const PathSegment& pathSegment) {
		return (pathSegment.remainingBounces > 0);
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

	startCpuTimer();
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if BVH
			int num_BVHnodes = (1 << (BVH_DEPTH + 1)) - 1;
			computeBVHIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_intersections
				, dev_BVHnodes,
				num_BVHnodes
				, dev_geoms
				);
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

#if MATSORT
		kernSetMatIDArr << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths,
			dev_intersections,
			dev_matIndicesI,
			dev_matIndicesP
			);

		dev_thrust_matIndicesI = thrust::device_ptr<int>(dev_matIndicesI);
		dev_thrust_intersections = thrust::device_ptr<ShadeableIntersection>(dev_intersections);

		dev_thrust_matIndicesP = thrust::device_ptr<int>(dev_matIndicesP);
		dev_thrust_paths = thrust::device_ptr<PathSegment>(dev_paths);

		// Wrap device vectors in thrust iterators for use with thrust.
		thrust::sort_by_key(dev_thrust_matIndicesI, dev_thrust_matIndicesI + num_paths, dev_thrust_intersections);
		thrust::sort_by_key(dev_thrust_matIndicesP, dev_thrust_matIndicesP + num_paths, dev_thrust_paths);

#endif
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

#if BVHDEBUG
			shadeDebugMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				num_paths,
				dev_intersections,
				dev_paths
				);
#else
			shadeAnyMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				num_paths,
				dev_intersections,
				dev_paths,
				dev_materials
				);
#endif

#if COMPACT
		PathSegment* compacted = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, has_bounces());
		num_paths = compacted - dev_paths;
#endif
		//printf("num paths: %i , depth: %i \n", num_paths, depth);
		if (num_paths == 0 || depth > traceDepth){
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}
	}
	num_paths = dev_path_end - dev_paths;
	printf("Iteration Done\n");
	endCpuTimer();
	printTime();
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
