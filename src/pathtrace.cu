#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <chrono>
#include <thread>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define SRT_SPP 1 // Anti- Aliasing - Square root of Sample per Pixel
#define USE_CACHE_PATH 0
#define USE_STREAM_COMPACTION 1
#define USE_RADIX_SORT 0
#define USE_BVIC 1 // bounding volume intersection culling
#define USE_OCTREE 0
#define USE_KDTREE 0
#define USE_BVH 1

#define NAIVE_ITERGRATOR 1
#define MIS_ITERGRATOR 0


#define DEBUG_NORMAL 0
#define DEBUG_UV 0
#define DEBUG_ROUGHNESS 0

#define ERRORCHECK 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

inline int ilog2ceil(int x) {
	return ilog2(x - 1) + 1;
}

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
static glm::vec3 * result_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static PathSegment * dev_paths_B = NULL;

static PathSegment * CACHE_paths = NULL;

static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

static int * bTerminatedforNaive = NULL;
static Geom * dev_lights = NULL;
static bool bdev_pathsUse = true;
static Triangle *dev_Triangles = NULL;
static Octree *dev_Octrees = NULL;

static Image * ImageHeader = NULL;
static glm::vec3 * imageDatas = NULL;

static KDtreeNodeForGPU *kdTreeForGPU = NULL;
static int *kdTreeTriangleIndexForGPU = NULL;

static BVHNodeForGPU *bvhTreeForGPU = NULL;
static int *bvhTreeTriangleIndexForGPU = NULL;

//static bool *dev_bTraversed = NULL;

//static IntersectedInvTransfromMat * dev_intersections = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y * SRT_SPP*SRT_SPP;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&result_image, pixelcount / (SRT_SPP*SRT_SPP) * sizeof(glm::vec3));
	cudaMemset(result_image, 0, pixelcount / (SRT_SPP*SRT_SPP) * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths_B, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


    // TODO: initialize any extra device memeory you need

	cudaMalloc(&bTerminatedforNaive, pixelcount* sizeof(int));
	cudaMemset(bTerminatedforNaive, 1, pixelcount * sizeof(int));
	

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&CACHE_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_Triangles, scene->trianglesInMesh.size() * sizeof(Triangle));
	cudaMemcpy(dev_Triangles, scene->trianglesInMesh.data(), scene->trianglesInMesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	

	cudaMalloc(&dev_Octrees, scene->octreeforMeshes.size() * sizeof(Octree));
	cudaMemcpy(dev_Octrees, scene->octreeforMeshes.data(), scene->octreeforMeshes.size() * sizeof(Octree), cudaMemcpyHostToDevice);

	//cudaMalloc(&dev_bTraversed, scene->octreeforMeshes.size() * sizeof(bool));
	//cudaMemcpy(dev_bTraversed, false, scene->octreeforMeshes.size() * sizeof(bool), cudaMemcpyHostToDevice);
	
	cudaMalloc(&ImageHeader, scene->Images.size() * sizeof(Image));
	cudaMemcpy(ImageHeader, scene->Images.data(), scene->Images.size() * sizeof(Image), cudaMemcpyHostToDevice);
	
	cudaMalloc(&imageDatas, scene->imageData.size() * sizeof(glm::vec3));
	cudaMemcpy(imageDatas, scene->imageData.data(), scene->imageData.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);	

	cudaMalloc(&kdTreeForGPU, scene->kdTreeForGPU.size() * sizeof(KDtreeNodeForGPU));
	cudaMemcpy(kdTreeForGPU, scene->kdTreeForGPU.data(), scene->kdTreeForGPU.size() * sizeof(KDtreeNodeForGPU), cudaMemcpyHostToDevice);

	cudaMalloc(&kdTreeTriangleIndexForGPU, scene->kdTreeTriangleIndexForGPU.size() * sizeof(int));
	cudaMemcpy(kdTreeTriangleIndexForGPU, scene->kdTreeTriangleIndexForGPU.data(), scene->kdTreeTriangleIndexForGPU.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&bvhTreeForGPU, scene->bvhTreeForGPU.size() * sizeof(BVHNodeForGPU));
	cudaMemcpy(bvhTreeForGPU, scene->bvhTreeForGPU.data(), scene->bvhTreeForGPU.size() * sizeof(BVHNodeForGPU), cudaMemcpyHostToDevice);

	cudaMalloc(&bvhTreeTriangleIndexForGPU, scene->bvhTreeTriangleIndexForGPU.size() * sizeof(int));
	cudaMemcpy(bvhTreeTriangleIndexForGPU, scene->bvhTreeTriangleIndexForGPU.data(), scene->bvhTreeTriangleIndexForGPU.size() * sizeof(int), cudaMemcpyHostToDevice);
	
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(result_image);
  	cudaFree(dev_paths);
	cudaFree(dev_paths_B);

  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);

    // TODO: clean up any extra device memory you created
	cudaFree(bTerminatedforNaive);
	cudaFree(dev_lights);
	cudaFree(CACHE_paths);
	cudaFree(dev_Triangles);
	cudaFree(dev_Octrees);
	//cudaFree(dev_bTraversed);

	cudaFree(ImageHeader);
	cudaFree(imageDatas);

	cudaFree(kdTreeForGPU);
	cudaFree(kdTreeTriangleIndexForGPU);
	
	cudaFree(bvhTreeForGPU);	
	cudaFree(bvhTreeTriangleIndexForGPU);
    checkCUDAError("pathtraceFree");
}

__host__ __device__ Point2f GenerateStratifiedSamples(int index_x, int index_y, float rn_x, float rn_y)
{	
	float invSqrtValX = 1.f / SRT_SPP;
	float invSqrtValY = 1.f / SRT_SPP;
	
	int x = index_x % SRT_SPP;
	int y = index_y % SRT_SPP;
	
	return Point2f((x + rn_x) *invSqrtValX + index_x/ SRT_SPP, (y + rn_y) * invSqrtValY + index_y/ SRT_SPP);
}


__host__ __device__ bool shadowIntersection(
	Ray &ray
	, Geom * geoms
	, Triangle * triangles
	, int geoms_size
	, ShadeableIntersection &intersections
	, Image* ImageHeader
	, glm::vec3* imageData
	, Material* materials
	, KDtreeNodeForGPU *pkdTreeForGPU
	, int *pkdTreeTriangleIndexForGPU
	, BVHNodeForGPU *pbvhTreeForGPU
	, int *pbvhTreeTriangleIndexForGPU
)
{

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	glm::vec2 uv;
	glm::mat4 transfromMat;
	glm::mat4 invtransfromMat;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	glm::vec2 tmp_uv;
	glm::mat4 tmp_transfromMat;
	glm::mat4 tmp_invtransfromMat;

	// naive parse through global geoms

	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];

#if USE_BVIC
		//Check AABB first
		if (BBIntersect(ray, geom.boundingBox, &t))
		{
#endif
			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);

			}
			else if (geom.type == PLANE)
			{
				t = planeIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);

			}
			else if (geom.type == MESH)
			{
#if USE_KDTREE
				t = traverseKDtree(ray, pkdTreeForGPU, geom.meshInfo.KDtreeID, geom, triangles, pkdTreeTriangleIndexForGPU, tmp_intersect, tmp_normal, tmp_uv, materials[geom.materialid].normalTexID, ImageHeader, imageData);
#elif USE_BVH
				t = traverseBVHtree(ray, pbvhTreeForGPU, geom.meshInfo.BVHtreeID, geom, triangles, pbvhTreeTriangleIndexForGPU, tmp_intersect, tmp_normal, tmp_uv, materials[geom.materialid].normalTexID, ImageHeader, imageData);
#else

				int size = geom.meshInfo.triangleBeginIndex + geom.meshInfo.size;

				for (int k = geom.meshInfo.triangleBeginIndex; k < size; k++)
				{
					t = triangleIntersectionTest(geom, triangles[k], ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);

					if (t > 0.0f && t_min > t)
					{
						t_min = t;
						hit_geom_index = i;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
						uv = tmp_uv;
					}
				}
#endif

			}
#if USE_BVIC
		}
		else
			t = -1.0f;
#endif
	
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
		intersections.t = -1.0f;
		return false;
	}
	else
	{
		//float RayEpsilon = 0.000005f;
		Geom *hitGeom = &geoms[hit_geom_index];
		//The ray hits something
		intersections.t = t_min;
		intersections.geomId = hitGeom->ID;// hit_geom_index;
		intersections.materialId = hitGeom->materialid;
		intersections.surfaceNormal = normal;

		intersections.ratationMat = hitGeom->transform;
		//intersections.ratationMat = hitGeom->rotationMat;
		intersections.IntersectedInvTransfrom = hitGeom->inverseTransform;
		intersections.intersectPoint = intersect_point;

		/*
		Vector3f originOffset = normal * ShadowEpsilon;

		if (glm::dot(ray.direction, normal) < 0.0f)
			originOffset = -originOffset;

		intersections.intersectPoint += originOffset;
		*/
		

		return true;
	}
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

	if (x < cam.resolution.x * SRT_SPP && y < cam.resolution.y * SRT_SPP)
	{
		int index = x + (y * cam.resolution.x * SRT_SPP);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.accumulatedColor = glm::vec3(0.0f);
		segment.throughputColor = glm::vec3(1.0f);
		segment.specularBounce = false;

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		//thrust::uniform_real_distribution<float> u02(0, 1);

		// Antialiasing
		// GenerateStratifiedSamples
		Point2f sampledCoords = GenerateStratifiedSamples(x, y, u01(rng), u01(rng));

		segment.ray.direction = glm::normalize(cam.view	- cam.right * cam.pixelLength.x * (sampledCoords.x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (sampledCoords.y - (float)cam.resolution.y * 0.5f)
			);

		
		//Depth of Field
		if (cam.lensRadious > 0.0f)
		{
			//World to Camera
			//glm::mat4 ViewMat = glm::lookAt( cam.position, cam.lookAt, cam.up);

			
			glm::vec3  f = glm::normalize(cam.lookAt - cam.position);
			glm::vec3  s = glm::normalize(glm::cross(f, cam.up));
			glm::vec3  u = glm::cross(s, f);

			glm::mat4 ViewMat;
			ViewMat[0][0] = s.x;
			ViewMat[1][0] = s.y;
			ViewMat[2][0] = s.z;
			ViewMat[3][0] = -glm::dot(s, cam.position);

			ViewMat[0][1] = u.x;
			ViewMat[1][1] = u.y;
			ViewMat[2][1] = u.z;
			ViewMat[3][1] = -glm::dot(u, cam.position);

			ViewMat[0][2] = -f.x;
			ViewMat[1][2] = -f.y;
			ViewMat[2][2] = -f.z;
			ViewMat[3][2] = glm::dot(f, cam.position);			
			
			ViewMat[0][3] = 0.0f;
			ViewMat[1][3] = 0.0f;
			ViewMat[2][3] = 0.0f;
			ViewMat[3][3] = 1.0f;

			glm::vec4 ori = ViewMat * glm::vec4(segment.ray.origin, 1.0f);
			glm::vec4 dir = ViewMat * glm::vec4(segment.ray.direction, 0.0f);

			//glm::vec4 ori = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
			//glm::vec4 dir = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);

			Ray ray;
			ray.origin = glm::vec3(ori);
			ray.direction = glm::vec3(dir);
				

			//Sample point on lens
			//Point3f pLens = cam.lensRadious * squareToDiskConcentric(glm::vec2( sampledCoords.x / (cam.resolution.x * SRT_SPP), sampledCoords.y / (cam.resolution.y * SRT_SPP)));
			Point3f pLens = cam.lensRadious * squareToDiskUniform(glm::vec2(u01(rng), u01(rng)));
			//Compute point on plane of focus
			float ft = glm::abs(cam.focalDistance / ray.direction.z);
			Point3f pFocus = ray.direction *ft + ray.origin;

			ray.origin = pLens;
			ray.direction = glm::normalize(pFocus - ray.origin);

			
			//Camera to World
			glm::mat4 InvViewMat = glm::inverse(ViewMat);
			ori = InvViewMat * glm::vec4(ray.origin, 1.0f);
			dir = InvViewMat * glm::vec4(ray.direction, 0.0f);

			segment.ray.origin = glm::vec3(ori);
			segment.ray.direction = glm::vec3(dir);
			
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
	, Triangle * triangles
	, Octree * octreees	
	, int octree_size
	, int geoms_size
	, ShadeableIntersection * intersections
	, Image* ImageHeader
	, glm::vec3* imageData
	, Material* materials
	, KDtreeNodeForGPU *pkdTreeForGPU
	, int *pkdTreeTriangleIndexForGPU
	, BVHNodeForGPU *pbvhTreeForGPU
	, int *pbvhTreeTriangleIndexForGPU
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::mat4 transfromMat;
		glm::mat4 invtransfromMat;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

		glm::mat4 tmp_transfromMat;
		glm::mat4 tmp_invtransfromMat;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];			

#if USE_BVIC
			//Check AABB first
			if (BBIntersect(pathSegment.ray, geom.boundingBox, &t))
			{
#endif
				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);
				}
				// TODO: add more intersection tests here... triangle? metaball? CSG?
				else if (geom.type == PLANE)
				{
					t = planeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);
				}
				else if (geom.type == MESH)
				{
#if USE_KDTREE
					t = traverseKDtree(pathSegment.ray, pkdTreeForGPU, geom.meshInfo.KDtreeID, geom, triangles, pkdTreeTriangleIndexForGPU, tmp_intersect, tmp_normal, tmp_uv, materials[geom.materialid].normalTexID, ImageHeader, imageData);
#elif USE_BVH
					t = traverseBVHtree(pathSegment.ray, pbvhTreeForGPU, geom.meshInfo.BVHtreeID, geom, triangles, pbvhTreeTriangleIndexForGPU, tmp_intersect, tmp_normal, tmp_uv, materials[geom.materialid].normalTexID, ImageHeader, imageData);
#else
					int size = geom.meshInfo.triangleBeginIndex + geom.meshInfo.size;
				
					for (int k = geom.meshInfo.triangleBeginIndex; k < size; k++)
					{
						t = triangleIntersectionTest(geom, triangles[k], pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);

						if (t > 0.0f && t_min > t)
						{
							t_min = t;
							hit_geom_index = i;
							intersect_point = tmp_intersect;
							normal = tmp_normal;
							uv = tmp_uv;
						}
					}
#endif
				}			
				else if (geom.type == TRIANGLE)
				{
					t = triangleIntersectionTest(geom, triangles[0], pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, materials[geom.materialid].normalTexID, ImageHeader, imageData);
				}					
#if USE_BVIC
			}
			else
				t = -1.0f;
#endif
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				uv = tmp_uv;
			}
		}

		ShadeableIntersection &SI = intersections[path_index];

		if (hit_geom_index == -1)
		{
			SI.t = -1.0f;
		}
		else
		{
			//float RayEpsilon = 0.000005f;
			Geom *hitGeom = &geoms[hit_geom_index];
			//The ray hits something
			SI.t = t_min;
			SI.geomId = hit_geom_index;
			SI.materialId = hitGeom->materialid;
			SI.surfaceNormal = normal;

			SI.ratationMat = hitGeom->rotationMat;
			SI.IntersectedInvTransfrom = hitGeom->inverseTransform;

			SI.intersectPoint = intersect_point;
			SI.uv = uv;		
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
        pathSegments[idx].throughputColor *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].throughputColor *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].throughputColor *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].throughputColor = glm::vec3(0.0f);
    }
  }
}

__global__ void Shading(
	int iter
	, int depth
	, int maxDepth
	, int num_paths
	, int num_geoms
	, Geom * geoms
	, int num_lights
	, Geom * lights
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, Image * ImageHeader
	, glm::vec3 * ImageData
	, int envMapID
	, int * terminated
	, Triangle * trianglesData
	, KDtreeNodeForGPU *pkdTreeForGPU
	, int *pkdTreeTriangleIndexForGPU
	, BVHNodeForGPU *pbvhTreeForGPU
	, int *pbvhTreeTriangleIndexForGPU
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	PathSegment &pathSegment = pathSegments[idx];

	if (idx < num_paths)
	{
		int &Terminator = terminated[idx];

		if (Terminator == 0)
		{
			return;
		}

		if (maxDepth <= depth)
		{
			Terminator = 0;
			return;
		}

		ShadeableIntersection &intersection = shadeableIntersections[idx];

		//Hit something
		if (intersection.t > 0.0f)
		{

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			//Hit the light
			if (material.emittance > 0.0f)
			{
				Terminator = 0;
				pathSegment.accumulatedColor += pathSegment.throughputColor * (materialColor * material.emittance);
				return;
			}

			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else
			{				
				glm::vec3 woW = -pathSegment.ray.direction;				
				Vector3f wiW;

#if DEBUG_UV
				Terminator = 0;
				pathSegment.accumulatedColor = glm::vec3(intersection.uv, 0.0f);
				return;
#endif
				
#if DEBUG_NORMAL
				Terminator = 0;
				pathSegment.accumulatedColor = (intersection.surfaceNormal + glm::vec3(1.0f))*0.5f;
				return;
#endif

				if (material.diffuseTexID >= 0)
				{
					Image ih = ImageHeader[material.diffuseTexID];
					glm::vec3 diffuseColor = getTextColor(ih.width, ih.height, ih.beginIndex, intersection.uv, ImageData);

					material.color *= diffuseColor;
				}

				if (material.specularTexID >= 0)
				{
					Image ih = ImageHeader[material.specularTexID];
					glm::vec3 specColor = getTextColor(ih.width, ih.height, ih.beginIndex, intersection.uv, ImageData);

					material.specular.color *= specColor;
				}

				if (material.roughnessTexID >= 0)
				{
					Image ih = ImageHeader[material.roughnessTexID];
					glm::vec3 roughness = getTextColor(ih.width, ih.height, ih.beginIndex, intersection.uv, ImageData);

					material.Roughness = glm::clamp(material.Roughness * roughness.x, 0.05f, 1.0f);
				}

#if NAIVE_ITERGRATOR

				Color3f f;

				//global illumination
				Float GlobalPdf = 0.0f;
							
				
				scatterRay(pathSegment, intersection.intersectPoint, woW, wiW, intersection.surfaceNormal, f, GlobalPdf, material, rng);


				if (IsBlack(f) || GlobalPdf <= 0.0f)
				{
					Terminator = 0;
					return;
				}

				pathSegment.throughputColor *= (f * glm::abs(glm::dot(wiW, intersection.surfaceNormal))) / GlobalPdf;
				Terminator = 1;
				return;
#endif

#if MIS_ITERGRATOR

				thrust::uniform_real_distribution<float> u01(0, 1);

				int ChosenLightIndex = (int)glm::min(u01(rng) * num_lights, (float)(num_lights - 1));
				float DLightPdf = 0.0f;
				float ScatteringPdf = 0.0f;

				Color3f Ld = Color3f(0.0f);

				PathSegment pathSegmentforDummy = pathSegment;
				
				Geom &selectedLight = lights[ChosenLightIndex];
							
				
				//Computing the direct lighting component (DL)
				
				if (!(pathSegment.specularBounce || material.hasRefractive > 0.0f))
				{
					Color3f f_DL;

					Color3f Li = Sample_Li(intersection, Point2f(u01(rng), u01(rng)), &wiW, &DLightPdf, selectedLight, materials);							

					GetFandPDF(intersection.intersectPoint, woW, wiW, intersection.surfaceNormal, f_DL, ScatteringPdf, material);

					if (ScatteringPdf < ShadowEpsilon)
						ScatteringPdf = 0.0f;

					f_DL = f_DL * glm::abs(glm::dot(wiW, intersection.surfaceNormal));					

					if (!IsBlack(Li) && !IsBlack(f_DL) && DLightPdf > 0)
					{
						//Shadow Test
						Ray ShadowRay = SpawnRay(intersection, wiW);
						ShadeableIntersection Shadowisect;

						if (shadowIntersection(ShadowRay, geoms, trianglesData, num_geoms, Shadowisect, ImageHeader, ImageData, materials, pkdTreeForGPU, pkdTreeTriangleIndexForGPU, pbvhTreeForGPU, pbvhTreeTriangleIndexForGPU))
						{
							if (Shadowisect.geomId == selectedLight.ID)
							{
								float weight = PowerHeuristic(1, DLightPdf, 1, ScatteringPdf);
								Ld += f_DL * Li * weight / DLightPdf;
							}
						}
					}
				}
				
				
				//Sample BSDF with multiple importance sampling				
				if (!(pathSegment.specularBounce || material.hasRefractive > 0.0f))
				{
					Color3f f_MIS;

					scatterRay(pathSegmentforDummy, intersection.intersectPoint, woW, wiW, intersection.surfaceNormal, f_MIS, ScatteringPdf, material, rng);
					f_MIS *= glm::abs(glm::dot(wiW, intersection.surfaceNormal));					

					if (ScatteringPdf < ShadowEpsilon)
						ScatteringPdf = 0.0f;

					if (!IsBlack(f_MIS) && ScatteringPdf > 0)
					{
						float weight = 1.0f;

						DLightPdf = Pdf_Li(intersection, wiW, selectedLight, material.normalTexID, ImageHeader, ImageData);
						weight = PowerHeuristic(1, ScatteringPdf, 1, DLightPdf);						

						if (DLightPdf > 0.0f)
						{
							Color3f Ldi(0.0f);
							Ray ShadowRay = SpawnRay(intersection, wiW);

							ShadeableIntersection Shadowisect;
							if (shadowIntersection(ShadowRay, geoms, trianglesData, num_geoms, Shadowisect, ImageHeader, ImageData, &material, pkdTreeForGPU, pkdTreeTriangleIndexForGPU, pbvhTreeForGPU, pbvhTreeTriangleIndexForGPU))
							{
								if (Shadowisect.geomId == selectedLight.ID)
								{
									Ldi = Li(intersection.surfaceNormal, -wiW, selectedLight, &material);
								}
							}
							else
								Ldi = Color3f(0.0f);

							if (!IsBlack(Ldi))
							{
								Ld += f_MIS * Ldi * weight / ScatteringPdf;
							}
						}
					}
				}
				

				Ld *= num_lights;

				pathSegment.accumulatedColor += Ld * pathSegment.throughputColor;
				//Terminator = 0;
				//return;

				//global illumination
				Float GlobalPdf = 0.0f;

				Color3f globalf;

				scatterRay(pathSegment, intersection.intersectPoint, woW, wiW, intersection.surfaceNormal, globalf, GlobalPdf, material, rng);

				if (GlobalPdf < ShadowEpsilon)
					GlobalPdf = 0.0f;

				if (IsBlack(globalf) || GlobalPdf <= 0.0f)
				{
					Terminator = 0;
					return;
				}


				pathSegment.throughputColor *= (globalf * glm::abs(glm::dot(wiW, intersection.surfaceNormal))) / GlobalPdf;
				


				if (material.Roughness == 0.0f)
					pathSegment.specularBounce = true;

				Terminator = 1;
				return;
#endif


#if DEBUG_ROUGHNESS
				Terminator = 0;
				pathSegment.accumulatedColor = Color3f(material.Roughness);
#endif

				
			}
		}
		else
		{
			Terminator = 0;
			
			if (envMapID >= 0)
			{
				pathSegment.accumulatedColor += pathSegment.throughputColor * InfiniteAreaLight_L(-pathSegment.ray.direction, ImageHeader[envMapID], ImageData);;
			}
			
			return;
		}
	}
}

//Extended version
__global__ void UpSweep(int *g_idata, int n, int lvloffset)
{
	extern __shared__ int temp[];
	int thid = threadIdx.x;
	int index = 2 * thid; //0 ~ 2047

	int offset = 1;

	int t1 = ((index + 1) + (blockIdx.x * n)) * lvloffset - 1;
	int t2 = ((index + 2) + (blockIdx.x * n)) * lvloffset - 1;

	temp[index] = g_idata[t1];
	temp[index + 1] = g_idata[t2];

	//Up-Sweep (Parallel Reduction)
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(index + 1) - 1;
			int bi = offset*(index + 2) - 1;


			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	__syncthreads();
	
	g_idata[t1] = temp[index];
	g_idata[t2] = temp[index + 1];
	
	
}

//Extended version
__global__ void DownSweep(int *g_idata, int n, int lvloffset)
{
	extern __shared__ int temp[];
	int thid = threadIdx.x;
	int index = 2 * thid;

	int offset = n;

	int t1 = ((index + 1) + (blockIdx.x * n)) * lvloffset - 1;
	int t2 = ((index + 2) + (blockIdx.x * n)) * lvloffset - 1;

	temp[index] = g_idata[t1];
	temp[index + 1] = g_idata[t2];

	//Down-Sweep
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;

		__syncthreads();

		if (thid < d)
		{

			int ai = offset*(index + 1) - 1;
			int bi = offset*(index + 2) - 1;


			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	g_idata[t1] = temp[index];
	g_idata[t2] = temp[index + 1];
}

__global__ void kernScatter(int n, PathSegment *odata, const PathSegment *idata, const int *bools, const int *indices, glm::vec3* image) {
	// TODO
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= n)
		return;

	if (bools[index] == 0)
	{
		PathSegment iterationPath = idata[index];
		image[iterationPath.pixelIndex] = iterationPath.accumulatedColor;
	}
	//If it is terminated, save it 
	else
	{
		odata[indices[index]] = idata[index];
		
	}
}

__global__ void kernEarray(int *g_e, ShadeableIntersection *g_idata, int n, int digit)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	g_e[index] = (g_idata[index].materialId >> digit & 0x01) ? 0 : 1;
	
}

__global__ void kernTarray(int *g_t, int *g_f, int *g_e, int n)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	int totalFalses = g_e[n - 1] + g_f[n - 1];

	g_t[index] = index - g_f[index] + totalFalses;

}

__global__ void kernDarray(int *g_d, int *e, int *f, int *t)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	g_d[index] = e[index] ? f[index] : t[index];

}

__global__ void kernLast(ShadeableIntersection *g_odata, ShadeableIntersection *g_idata, PathSegment *g_opathdata , PathSegment *g_ipathdata, int *d, int n)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= n)
		return;

	int newIndex = d[index];

	g_odata[newIndex] = g_idata[index];
	g_opathdata[newIndex] = g_ipathdata[index];
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];		
		image[iterationPath.pixelIndex] = iterationPath.accumulatedColor;
	}
}


__global__ void averageColor(int iter, glm::ivec2 Resolution, glm::vec3 * image, glm::vec3 * resultImage)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	glm::vec3 tempResult =  glm::vec3(0.0f, 0.0f, 0.0f);

	int resultIndex = x + (y * Resolution.x);

#if SRT_SPP > 1

	if (x < Resolution.x && y < Resolution.y)
	{
		
		
		for (int i = 0; i < SRT_SPP; i++)
		{
			for (int j = 0; j < SRT_SPP; j++)
			{
				tempResult += image[(x*SRT_SPP + i) + ((y*SRT_SPP + j) * Resolution.x * SRT_SPP)];
			}
		}
		
		tempResult /= SRT_SPP*SRT_SPP;
		
	}
#else
	    tempResult += image[resultIndex];
#endif

	//Denoise
	//glm::vec3 preAvg = resultImage[resultIndex] / (float) iter;	
	//tempResult = glm::max(glm::clamp(tempResult, preAvg - Color3f(30.0f), preAvg + Color3f(30.0f)), Color3f(0.0f));

	resultImage[resultIndex] += tempResult;
}

void radixScan(int n, ShadeableIntersection *idata, PathSegment *paths)
{
	ShadeableIntersection *g_idata;
	ShadeableIntersection *g_odata;

	PathSegment *g_ipathdata;
	PathSegment *g_opathdata;

	int *g_e;
	int *g_f;
	int *g_t;
	int *g_d;

	int level = ilog2ceil(n);
	int poweroftwosize = (int)pow(2, level);

	cudaMalloc((void**)&g_idata, poweroftwosize * sizeof(ShadeableIntersection));
	cudaMalloc((void**)&g_odata, poweroftwosize * sizeof(ShadeableIntersection));

	cudaMemcpy(g_idata, idata, sizeof(ShadeableIntersection) * n, cudaMemcpyDeviceToDevice);

	cudaMalloc((void**)&g_ipathdata, n * sizeof(PathSegment));
	cudaMalloc((void**)&g_opathdata, n * sizeof(PathSegment));

	cudaMemcpy(g_ipathdata, paths, sizeof(PathSegment) * n, cudaMemcpyDeviceToDevice);	

	cudaMalloc((void**)&g_e, poweroftwosize * sizeof(int));	
	cudaMalloc((void**)&g_f, poweroftwosize * sizeof(int));
	cudaMalloc((void**)&g_t, poweroftwosize * sizeof(int));
	cudaMalloc((void**)&g_d, poweroftwosize * sizeof(int));
	

	for (int i = 0; i < level; i++)
	{
		int blockSize = pow(2, level);
		blockSize = std::min(blockSize, 1024);
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

		kernEarray <<< fullBlocksPerGrid, blockSize >>>(g_e, g_idata, n, i);

		cudaMemcpy(g_f, g_e, sizeof(int) * poweroftwosize, cudaMemcpyDeviceToDevice);



		int blockSize2 = pow(2, level);
		blockSize2 = std::min(blockSize2, 2048);
		int blockCount = (n + blockSize2 - 1) / blockSize2;

		int offset = 1;

		do
		{
			dim3 fullBlocksPerGrid(blockCount);

			UpSweep <<< fullBlocksPerGrid, blockSize2 / 2, blockSize2 * sizeof(int) >> > (g_f, blockSize2, offset);

			if (blockCount == 1)
				blockCount = 0;
			else
			{
				blockSize2 = blockCount;
				blockCount = 1;

				int level = ilog2ceil(blockSize2);
				blockSize2 = std::min((int)pow(2, level), 2048);

				blockCount = (blockCount + blockSize2 - 1) / blockSize2;
				offset *= 2048;
			}
		}
		while (blockCount >= 1);

		// clear the last element		
		cudaMemset(&g_f[poweroftwosize - 1], 0, sizeof(int));

		blockCount = 1;
		
		do
		{
			dim3 fullBlocksPerGrid(blockCount);
			DownSweep <<< fullBlocksPerGrid, blockSize2 / 2, blockSize2 * sizeof(int) >> > (g_f, blockSize2, offset);
			blockCount *= blockSize2;
			blockSize2 = 2048;
			offset /= 2048;

		} while (blockCount < n);

		kernTarray << <fullBlocksPerGrid, blockSize >> >(g_t, g_f, g_e, n);
		kernDarray << <fullBlocksPerGrid, blockSize >> >(g_d, g_e, g_f, g_t);
		kernLast << <fullBlocksPerGrid, blockSize >> > (g_odata, g_idata, g_opathdata, g_ipathdata, g_d, n);		

		cudaMemcpy(g_idata, g_odata, sizeof(ShadeableIntersection) * n, cudaMemcpyDeviceToDevice);
		cudaMemcpy(g_ipathdata, g_opathdata, sizeof(PathSegment) * n, cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(idata, g_odata, n * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	cudaMemcpy(paths, g_opathdata, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

	cudaFree(g_idata);
	cudaFree(g_odata);
	cudaFree(g_ipathdata);
	cudaFree(g_opathdata);

	cudaFree(g_e);
	cudaFree(g_t);
	cudaFree(g_f);
	cudaFree(g_d);
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {

	int * bTerminated = NULL;
	int * bTerminated_A = NULL;

	int * btempTerminated = NULL;


    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y * SRT_SPP * SRT_SPP;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(32, 32);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x * SRT_SPP + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y * SRT_SPP + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 256;

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
	bdev_pathsUse = true;
	PathSegment* Using_dev_path = dev_paths;

#if USE_CACHE_PATH
	if (iter == 1)
	{
			generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, Using_dev_path);
			checkCUDAError("generate camera ray");
			cudaMemcpy(CACHE_paths, Using_dev_path, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	//re-use
	else
	{
		cudaMemcpy(dev_paths, CACHE_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		//Using_dev_path = CACHE_paths;
	}
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, Using_dev_path);
	checkCUDAError("generate camera ray");
#endif

 	int depth = 0;
	PathSegment* dev_path_end = Using_dev_path + pixelcount;
	int num_paths = dev_path_end - Using_dev_path;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
	
	cudaMemset(bTerminatedforNaive, 1, pixelcount * sizeof(int));

	while (!iterationComplete)
	{

		int* terminateBuffer = NULL;

		Using_dev_path = bdev_pathsUse ? dev_paths : dev_paths_B;
#if	USE_STREAM_COMPACTION

		int level = ilog2ceil(num_paths);
		int TWOnum_paths = (int)pow(2, level);

		cudaMalloc(&bTerminated, TWOnum_paths * sizeof(int));
		cudaMemset(bTerminated, 0, TWOnum_paths * sizeof(int));
		cudaMemset(bTerminated, 1, num_paths * sizeof(int));

		cudaMalloc(&bTerminated_A, TWOnum_paths * sizeof(int));
		//cudaMemset(bTerminated_A, 1, TWOnum_paths * sizeof(int));		
		btempTerminated = new int[TWOnum_paths];
		terminateBuffer = bTerminated;
#else
		
		terminateBuffer = bTerminatedforNaive;
		btempTerminated = new int[pixelcount];

#endif
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		//cudaMemset(dev_bTraversed, false, hst_scene->octreeforMeshes.size() * sizeof(bool));
		

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth
			, num_paths
			, Using_dev_path
			, dev_geoms
			, dev_Triangles
			, dev_Octrees
			, hst_scene->octreeforMeshes.size()
			, hst_scene->geoms.size()
			, dev_intersections
			, ImageHeader
			, imageDatas
			, dev_materials
			, kdTreeForGPU
		    , kdTreeTriangleIndexForGPU
			, bvhTreeForGPU
			, bvhTreeTriangleIndexForGPU
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;


#if USE_RADIX_SORT
		radixScan(num_paths, dev_intersections, Using_dev_path);
		checkCUDAError("radixScan");
#endif
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		Shading <<<numblocksPathSegmentTracing, blockSize1d>>> (
		iter,
		depth,
		traceDepth,
		num_paths,
		hst_scene->geoms.size(),
		dev_geoms,
		hst_scene->lights.size(),
		dev_lights,
		dev_intersections,
		Using_dev_path,
		dev_materials,
		ImageHeader,
		imageDatas,
		hst_scene->envMapId,
		terminateBuffer,
		dev_Triangles,
		kdTreeForGPU,
		kdTreeTriangleIndexForGPU,
		bvhTreeForGPU,
		bvhTreeTriangleIndexForGPU
		);
		checkCUDAError("Shading");

		//Stream compaction		
	
#if	USE_STREAM_COMPACTION
			//SCAN
			cudaMemcpy(bTerminated_A, terminateBuffer, TWOnum_paths * sizeof(int), cudaMemcpyDeviceToDevice);

			int blockSize = TWOnum_paths;
			int numofThreadsforSC = std::min(blockSize, 2048);

			int numofblocksforSC = (TWOnum_paths + numofThreadsforSC - 1) / numofThreadsforSC;

			dim3 numblocksPathStreamCompaction = numofblocksforSC;

			int lvloffset = 1;

			do
			{
				UpSweep << < numblocksPathStreamCompaction, numofThreadsforSC / 2, numofThreadsforSC * sizeof(int) >> > (bTerminated_A, numofThreadsforSC, lvloffset);



				if (numofblocksforSC == 1)
					numofblocksforSC = 0;
				else
				{
					numofThreadsforSC = numblocksPathStreamCompaction.x;


					int level = ilog2ceil(numofThreadsforSC);
					numofThreadsforSC = std::min((int)pow(2, level), 2048);

					numofblocksforSC = (numofblocksforSC + numofThreadsforSC - 1) / numofThreadsforSC;
					numblocksPathStreamCompaction = numofblocksforSC;

					lvloffset *= 2048;
				}
			} while (numofblocksforSC >= 1);

			// clear the last element
			//int Zero = 0;
			//cudaMemcpy(&bTerminated_A[TWOnum_paths - 1], &Zero, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(&bTerminated_A[TWOnum_paths - 1], 0, sizeof(int));

			numofblocksforSC = 1;

			do
			{
				dim3 numblocksPathStreamCompaction(numofblocksforSC);
				DownSweep << < numblocksPathStreamCompaction, numofThreadsforSC / 2, numofThreadsforSC * sizeof(int) >> > (bTerminated_A, numofThreadsforSC, lvloffset);

				numofblocksforSC *= numofThreadsforSC;
				lvloffset /= 2048;
				numofThreadsforSC = 2048;

			} while (numofblocksforSC < num_paths);

			//cudaMemset(&bTerminated_A[TWOnum_paths - 1], 0, sizeof(int));

			//SCATTER & FINAL GATHER
			kernScatter << <numblocksPathSegmentTracing, blockSize1d >> >(num_paths, bdev_pathsUse ? dev_paths_B : dev_paths, bdev_pathsUse ? dev_paths : dev_paths_B, terminateBuffer, bTerminated_A, dev_image);

			bdev_pathsUse = !bdev_pathsUse;

			/*
			cudaMemcpy(btempTerminated, bTerminated_A, TWOnum_paths * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i < TWOnum_paths; i++)
			{
				//if (btempTerminated[i] > 0)
				//{
				cout << "[" << i << "]" << ":" << btempTerminated[i] << ", ";

				//}

				//std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}

			cout << endl << endl;
			*/
			//std::this_thread::sleep_for(std::chrono::milliseconds(20000));

			int counter;
			cudaMemcpy(&counter, &bTerminated_A[TWOnum_paths - 1], sizeof(int), cudaMemcpyDeviceToHost);

			if (counter == 0)
				iterationComplete = true;
			else
				num_paths = counter;


			

			cudaFree(bTerminated);
			cudaFree(bTerminated_A);

			checkCUDAError("STREAM_COMPACTION");
#else
			//Naive
			
			cudaMemcpy(btempTerminated, terminateBuffer, pixelcount * sizeof(int), cudaMemcpyDeviceToHost);

			int xy = 0;

			for (int i = 0; i < pixelcount; i++)
			{
				if (btempTerminated[i] != 0)
				{
					iterationComplete = false;
					break;
				}
				else if (i == pixelcount - 1 && btempTerminated[i] == 0)
					iterationComplete = true;
			}

			

			/*
			for (int i = 0; i < pixelcount; i++)
			{
				if (btempTerminated[i] != 0)
					xy++;
			}

			if(xy > 0)
				iterationComplete = false;
			else
				iterationComplete = true;
			*/
			/*
			for (int i = 0; i < pixelcount; i++)
			{
				if(btempTerminated[i] != 0)
				cout << "[" << i << "]" << ":" << btempTerminated[i] << ", ";
			}

			cout << endl;
			*/
			
		
#endif 		


		delete[] btempTerminated;
		


		



	}
	checkCUDAError("itergrator");

	
	
    // Assemble this iteration and apply it to the image
#if !USE_STREAM_COMPACTION
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);
	checkCUDAError("finalGather");
#endif 

	
	
	
    //dim3 numBlocksResultPixels = (pixelcount / SPP + blockSize1d - 1) / blockSize1d;

	const dim3 blocksPerResultGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	//cudaMemset(result_image, 0, pixelcount / (SRT_SPP*SRT_SPP) * sizeof(glm::vec3));
	
	averageColor<<<blocksPerResultGrid2d, blockSize2d >>>(iter, cam.resolution, dev_image, result_image);
	checkCUDAError("averageColor");

    // Send results to OpenGL buffer for rendering

	dim3 blocksPerResolutionGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    sendImageToPBO<<<blocksPerResultGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, result_image);
	checkCUDAError("sendImageToPBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), result_image,
            pixelcount / (SRT_SPP*SRT_SPP) * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
