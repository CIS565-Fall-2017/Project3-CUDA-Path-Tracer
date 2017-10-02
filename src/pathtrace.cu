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
#define OCTREE 1

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

__global__ void kernRearrangeGeomToOctree(
	const int* octGeomIdx,
	Geom* geomsOld,
	Geom* geomsNew,
	int num_geoms)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= num_geoms) return;

	geomsNew[idx] = geomsOld[octGeomIdx[idx]];
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
// ...

static OctreeNodeGPU* dev_octree = NULL;
static glm::vec3 sceneBounds;

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

	checkCUDAError("pathtraceInit");

#ifdef OCTREE
	OctreeBuilder octree = OctreeBuilder();
	octree.buildFromScene(scene);
	octree.buildGPUfromCPU();
	sceneBounds = octree.sceneHalfEdgeSize();

	cudaMalloc(&dev_octree, octree.allGPUNodes.size() * sizeof(OctreeNodeGPU));
	cudaMemcpy(dev_octree, octree.allGPUNodes.data(), octree.allGPUNodes.size() * sizeof(OctreeNodeGPU), cudaMemcpyHostToDevice);
	checkCUDAError("copying octree data");

	Geom* dev_geoms_temp;
	cudaMalloc(&dev_geoms_temp, scene->geoms.size() * sizeof(Geom));
	checkCUDAError("initializing temp geom buffer");
	int* dev_geomIdx_temp;
	cudaMalloc(&dev_geomIdx_temp, octree.octreeOrderGeomIDX.size() * sizeof(int));
	checkCUDAError("initializing temp geom idx buffer");
	cudaMemcpy(dev_geomIdx_temp, octree.octreeOrderGeomIDX.data(), octree.octreeOrderGeomIDX.size() * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("copying temp geometry idx");

	const int blockSize = 128;
	dim3 numblocksGeomTransfer = (scene->geoms.size() + blockSize - 1) / blockSize;
	kernRearrangeGeomToOctree << <numblocksGeomTransfer, blockSize >> >(dev_geomIdx_temp, dev_geoms, dev_geoms_temp, scene->geoms.size());
	checkCUDAError("sorting geometry by octree ordering");

	Geom* temp = dev_geoms;
	dev_geoms = dev_geoms_temp;
	dev_geoms_temp = temp;

	cudaFree(dev_geomIdx_temp);
	cudaFree(dev_geoms_temp);
	checkCUDAError("ending OCtree setup");

#endif




}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
#ifdef  OCTREE
	cudaFree(dev_octree);
#endif //  OCTREE


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

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, ((x + iter) % y) * ((y + iter / 3) % x), 0);
	thrust::uniform_real_distribution<float> unif(-0.5, 0.5);
	thrust::uniform_real_distribution<float> u01(0, 1);


	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		glm::vec3 jitter = cam.right * cam.pixelLength.x * unif(rng) + cam.up * cam.pixelLength.y * unif(rng);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			+ jitter
			);

		float lensR = sqrt(u01(rng)) * cam.lensRadius;
		float lensT = u01(rng) * 2.0f * PI;

		glm::vec3 lensO = lensR * cos(lensT) * cam.right + lensR * sin(lensT) * cam.up + segment.ray.origin;
		segment.ray.direction = glm::normalize(segment.ray.direction * cam.focusDistance + segment.ray.origin - lensO);
		segment.ray.origin = lensO;

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersectionsOctree(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	Geom* geoms,
	ShadeableIntersection * intersections,
	OctreeNodeGPU * octree,
	glm::vec3 sceneHalfEdgeSize
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 temp_intersect;
	glm::vec3 temp_normal;

	// this will have the remaining number of children to check at each level in the octree
	int remainingChildrenStack[12];
	bool testedGeometryStack[12];
	// first: get the number of children of root, push it to stack
	int stackDepthPtr = 0;
	testedGeometryStack[stackDepthPtr] = false;
	remainingChildrenStack[stackDepthPtr] = octree[0].childEndIdx - octree[0].childStartIdx;

	OctreeNodeGPU current = octree[0];
	PathSegment pathSegment = pathSegments[path_index];

	// if doesn't hit the root, return immediately
	t = aabbIntersectionTest(current.center, sceneHalfEdgeSize, pathSegment.ray, temp_intersect, temp_normal, outside);
	if (t < 0.0f) {
		intersections[path_index].t = -1.0f;
		return;
	}

	do {

		// test intersection of geometries in this node.
		// unfortunately needs to be part of loop, but only done once per node
		if (!testedGeometryStack[stackDepthPtr]) {
			for (int i = current.eltStartIdx; i < current.eltEndIdx; i++) {
				Geom & geom = geoms[i];

				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray, temp_intersect, temp_normal, outside);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray, temp_intersect, temp_normal, outside);
				}
				else if (geom.type == TRIANGLE) {
					t = triangleIntersectionTest(geom, pathSegment.ray, temp_intersect, temp_normal, outside);
				}

				// Compute the minimum t from the intersection tests to determine what
				// scene geometry object was hit first.
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = i;
					intersect_point = temp_intersect;
					normal = temp_normal;
				}
			}
			testedGeometryStack[stackDepthPtr] = true; // mark as finished
		}


		// see if there are any remaining children
		if (remainingChildrenStack[stackDepthPtr] == 0) {

			// back up stack, set current to its parent
			stackDepthPtr -= 1;
			int parentIdx = current.parentIdx;
			if (parentIdx > -1) current = octree[parentIdx]; // edge case, root
			else break;
		}
		else {
			// test intersection against next child
			int childIdx = current.childEndIdx - remainingChildrenStack[stackDepthPtr];
			OctreeNodeGPU child = octree[childIdx];
			int depthDivisor = 1 << (child.depth);
			glm::vec3 childHalfEdgeSize = sceneHalfEdgeSize / glm::vec3(depthDivisor);
			glm::vec3 childCenter = child.center;
			// decrement remaining ptr on stack
			remainingChildrenStack[stackDepthPtr] -= 1;

			// try to intersect the bounding box of this octree node
			float octT = aabbIntersectionTest(childCenter, childHalfEdgeSize, pathSegment.ray, temp_intersect, temp_normal, outside);
			
			if (octT > 0.0f) {// && t_min > octT) {
				// push info to stack, set current to child
				stackDepthPtr++;
				current = child;
				testedGeometryStack[stackDepthPtr] = false;
				remainingChildrenStack[stackDepthPtr] = current.childEndIdx - current.childStartIdx;
			}
		}	
	} while (stackDepthPtr > -1);

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

__device__ glm::vec3 schlickFresnel(float VdotH, glm::vec3 reflect0, glm::vec3 reflect90)
{
	VdotH = min(1.0f, max(0.0f, 1.0f - VdotH));
	return reflect0 + (reflect90 - reflect0) * pow(VdotH, 5.0f);
}

__global__ void shadeMaterialBasic(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	
	ShadeableIntersection isx = shadeableIntersections[idx];
	if (isx.t > 0.0f) {
		if (pathSegments[idx].remainingBounces <= 0) return;
		int hash = idx + pathSegments[idx].remainingBounces + iter;
		hash = (hash << 13) ^ hash;
		hash = (hash << 7 + hash) ^ hash;
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, hash, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		Material mat = materials[isx.materialId];
		glm::vec3 isxPoint = isx.t * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
		pathSegments[idx].ray.origin = isxPoint + isx.surfaceNormal * 0.0001f;
		glm::vec3 rayOut = glm::normalize(pathSegments[idx].ray.direction); // from eye to intersection

		if (mat.emittance > 0.0f) {
			pathSegments[idx].color *= (mat.emittance * mat.color);
			pathSegments[idx].remainingBounces = 0;
		}
		else {
			
			if (0) {// temp disabled(mat.hasRefractive) {
				// calculate base fresnel to decide whether to refract or reflect
				bool external = glm::dot(isx.surfaceNormal, rayOut) < 0.0f;
				float NdotV = glm::abs(glm::dot(-rayOut, isx.surfaceNormal));
				glm::vec3 f0 = glm::vec3(0.04f);
				glm::vec3 f90 = glm::vec3(1.0f);
				glm::vec3 F = schlickFresnel(NdotV, f0, f90);
				float chance = u01(rng);
				if (0){// (chance < F.x && external) {
					// reflected ray
					glm::vec3 reflected = glm::reflect(rayOut, isx.surfaceNormal);
					pathSegments[idx].color *= (mat.specular.color / F.x); // divide by probability
					pathSegments[idx].remainingBounces = pathSegments[idx].remainingBounces - 1;
					pathSegments[idx].ray.direction = reflected;
				}
				else {
					// refracted ray		
					float eta = external ? mat.indexOfRefraction : 1.0f / mat.indexOfRefraction;
					glm::vec3 refracted = glm::refract(rayOut, (external? 1.0f : -1.0f) * isx.surfaceNormal, eta);
					refracted = glm::normalize(refracted);
					pathSegments[idx].color *= (external ? mat.color : glm::vec3(1.0f));
					//pathSegments[idx].color *= abs(glm::dot(refracted, isx.surfaceNormal));
					pathSegments[idx].remainingBounces -= external ? 1 : 0;
					pathSegments[idx].ray.direction = refracted; //  totalInternal ? rayOut : refracted;
					pathSegments[idx].ray.origin = isxPoint + isx.surfaceNormal * (external? -0.0001f : 0.0001f);
					
				}
			}
			if (mat.hasReflective) {
				glm::vec3 reflected = glm::reflect(pathSegments[idx].ray.direction, isx.surfaceNormal);
				pathSegments[idx].color *= mat.specular.color;
				pathSegments[idx].remainingBounces = pathSegments[idx].remainingBounces - 1;
				pathSegments[idx].ray.direction = reflected;
			}
			else {
				// lambertian color
				glm::vec3 lambertIn = calculateRandomDirectionInHemisphere(isx.surfaceNormal, rng);	
				pathSegments[idx].ray.direction = lambertIn;		
				pathSegments[idx].color *= mat.color;	
				pathSegments[idx].remainingBounces = (glm::dot(pathSegments[idx].color, glm::vec3(1.0f)) < 0.0001) ? 0.0 : pathSegments[idx].remainingBounces - 1;
				
			}

			if (pathSegments[idx].remainingBounces == 0) pathSegments[idx].color *= glm::vec3(0.0);
			//pathSegments[idx].color = glm::abs(pathSegments[idx].ray.direction);
		}
	}
	else {
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = 0;
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

struct is_live
{
	__host__ __device__
		bool operator()(const PathSegment &x)
	{
		return (x.remainingBounces) > 0;
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
	int active_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;

	for (int i = 0; i < traceDepth && !iterationComplete; i++) {
		//for (int i = 0; i < 1; i++) {
	//while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		dim3 numblocksPathSegmentTracing = (active_paths + blockSize1d - 1) / blockSize1d;

#ifdef OCTREE
		// tracing
		//int depth,
			//int num_paths,
			//PathSegment* pathSegments,
			//Geom* geoms,
			//ShadeableIntersection * intersections,
			//OctreeNodeGPU * octree,
			//glm::vec3 sceneHalfEdgeSize
		computeIntersectionsOctree << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			active_paths,
			dev_paths,
			dev_geoms,
			dev_intersections,
			dev_octree,
			sceneBounds
			);


#else
		// tracing

		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, active_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
#endif // OCTREE


		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		shadeMaterialBasic << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			active_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);

		PathSegment* partMiddle = thrust::partition(thrust::device, dev_paths, dev_paths + active_paths, is_live());
		active_paths = partMiddle - dev_paths;
		iterationComplete = active_paths <= 0;
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
