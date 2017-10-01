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
#include "build\src\FresnelDielectric.h"


#define lensRadius 1.f
#define focalDistance 2.f
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
//static PathSegment* dev_finalSeg = NULL;
//static bool* streamFlag = NULL;
static int* compactSteamsIn= NULL;
static int* materialKey = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	//cudaMalloc(&dev_finalSeg, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

	cudaMalloc(&compactSteamsIn, pixelcount * sizeof(int));
	//cudaMalloc(&streamFlag, sizeof(bool)*pixelcount);
	cudaMalloc(&materialKey, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

	//cudaFree(dev_finalSeg);
	cudaFree(compactSteamsIn);
	//cudaFree(streamFlag);
	cudaFree(materialKey);
    checkCUDAError("pathtraceFree");
}

__device__ glm::vec3 squareToDiskConcentric(glm::vec2 sample)
{
	float phi, r, u, v;
	float a = 2 * sample[0] - 1;
	float b = 2 * sample[1] - 1;

	if (a>-b)
	{
		if (a>b)
		{
			r = a;
			phi = (PI / 4)*(b / a);
		}
		else
		{
			r = b;
			phi = (PI / 4)*(2 - (a / b));
		}
	}
	else
	{
		if (a<b)
		{
			r = -a;
			phi = (PI / 4)*(4 + (b / a));
		}
		else
		{
			r = -b;
			if (b != 0)
			{
				phi = (PI / 4)*(6 - (a / b));
			}
			else
			{
				phi = 0;
			}
		}
	}

	u = r*cos(phi);
	v = r*sin(phi);
	return glm::vec3(u, v, 0);
}

__device__ void RealisticCamera(PathSegment& pathsegment, thrust::default_random_engine &rng)
{
	glm::vec3 pinholeRayOri = pathsegment.ray.origin;
	glm::vec3 pinholeRayDir = pathsegment.ray.direction;
	glm::vec3 rayOrigin = glm::vec3(0.f);
	glm::vec3 rayDirection = glm::vec3(0.f);

	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::uniform_real_distribution<float> u02(0, 1);
	float samplex = u01(rng);
	float sampley = u02(rng);

	if (lensRadius > 0)
	{
		glm::vec3 pLens = lensRadius * squareToDiskConcentric(glm::vec2(samplex,sampley));

		float ft = (focalDistance - pinholeRayOri.z) / pinholeRayDir.z;
		glm::vec3 pFocus = pinholeRayOri + pinholeRayDir * ft;


		rayOrigin = pinholeRayOri + pLens;
		rayDirection = glm::normalize(pFocus - rayOrigin);
	}
	pathsegment.ray.origin = rayOrigin;
	pathsegment.ray.direction = rayDirection;

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
		
		//TODO:special code for realistic camera
		//Realistic Camera Part
		/*thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		RealisticCamera(pathSegments[index], rng);*/
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
		}
	}
}

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	,int depth
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths)
	{
	    if ((pathSegments[idx].remainingBounces == 0) && (materials[shadeableIntersections[idx].materialId].emittance == 0.f))
		{
			pathSegments[idx].color = glm::vec3(0.f);
			//return;
		}
		else if((pathSegments[idx].remainingBounces == 0) &&(materials[shadeableIntersections[idx].materialId].emittance >= 0.f)){
			return;
		}
		else {
			int bounceDepth = depth - pathSegments[idx].remainingBounces;
			ShadeableIntersection intersection = shadeableIntersections[idx];
			if (intersection.t > 0.0f) { // if the intersection exists...

				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, bounceDepth);
				thrust::uniform_real_distribution<float> u01(0, 1);

				Material materialS = materials[intersection.materialId];
				glm::vec3 materialColor = materialS.color;

				// If the material indicates that the object was a light, "light" the ray
				if (materialS.emittance > 0.0f) {
					pathSegments[idx].color *= (materialColor * materialS.emittance);
					pathSegments[idx].remainingBounces = 0;
				}
				else {

					glm::vec3 lastIntersectionPoint = pathSegments[idx].ray.origin;
					glm::vec3 lastIntersectionDir = pathSegments[idx].ray.direction;
					glm::vec3 intersectionPoint = lastIntersectionPoint + lastIntersectionDir*intersection.t;
					scatterRay(pathSegments[idx], intersectionPoint, intersection.surfaceNormal, materialS, rng);
					//pathSegments[idx].remainingBounces--;
				}
			}
			else {
				pathSegments[idx].color = glm::vec3(0.0f);
				pathSegments[idx].remainingBounces = 0;
			}
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

  if (glm::length(pathSegments[idx].color - glm::vec3(0.f)) <= FLT_EPSILON)
  {
	  pathSegments[idx].remainingBounces = 0;
	  return;
  }

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
		pathSegments[idx].remainingBounces=0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        //pathSegments[idx].color *= u01(rng); // apply some noise because why not

		  glm::vec3 lastIntersectionPoint = pathSegments[idx].ray.origin;
		  glm::vec3 lastIntersectionDir = pathSegments[idx].ray.direction;
		  glm::vec3 intersectionPoint = lastIntersectionPoint + lastIntersectionDir*intersection.t;
		  scatterRay(pathSegments[idx], intersectionPoint, intersection.surfaceNormal, material, rng);
		  pathSegments[idx].remainingBounces--;
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
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void BouncesLeft(PathSegment * pathSegments, int* steamCompactIn, int pixelCount)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < pixelCount)
	{
		PathSegment thisPath = pathSegments[index];
		steamCompactIn[index] = thisPath.remainingBounces;
	}
}

__device__ int DirectShadowIntersection(
	PathSegment & pathSegment
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection & intersection)
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

		
		//material 0 means hit the light source
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
		}
				
}

//TODO My own Path tracer DirectLighting pa
__global__ void DirectLightingIntegrator(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, Geom * geoms
    , int lightCount
    , int depth
    , int geoSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		Material materialIsc = materials[intersection.materialId];
		int remainBounce = depth - pathSegments[idx].remainingBounces;

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, remainBounce);
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::uniform_real_distribution<float> u02(0, 1);
		thrust::uniform_real_distribution<float> u03(0, 1);

		//hit something
		if (intersection.t > 0.f)
		{
			//hit light source
			if (materialIsc.emittance > 0.f)
			{
				pathSegments[idx].color = materialIsc.color;
			}
			//other situations 
			else
			{
				ShadeableIntersection shadowIntersection;
				glm::vec3 lastRayOrigin = pathSegments[idx].ray.origin;
				glm::vec3 lastRayDir = pathSegments[idx].ray.direction;
				glm::vec3 intersectionPoint = lastRayOrigin + glm::normalize(lastRayDir)*intersection.t;
				
				PathSegment shadowSegment;
				PathSegment lastSegment = pathSegments[idx];

				float sampleX = (u01(rng)-0.5)*2.f;
				float sampleY = (u02(rng)-0.5)*2.f;
				float sampleZ = (u03(rng)-0.5)*2.f;

				glm::vec3 lightPoint = geoms[0].translation + glm::vec3(sampleX*geoms[0].scale.x*0.5, sampleY*geoms[0].scale.y*0.5, sampleZ*geoms[0].scale.z*0.5);
				shadowSegment.ray.origin = intersectionPoint;
				shadowSegment.ray.direction = glm::normalize(lightPoint - intersectionPoint);
				shadowSegment.ray.origin += intersection.surfaceNormal*0.013f;
				shadowSegment.pixelIndex = lastSegment.pixelIndex;
				shadowSegment.color = lastSegment.color;

				if (glm::dot(shadowSegment.ray.direction, intersection.surfaceNormal) < 0.f)
				{
					pathSegments[idx].color = glm::vec3(0.f);
					return;
				}

				DirectShadowIntersection(shadowSegment, geoms, geoSize, shadowIntersection);
				Material shadowMaterial = materials[shadowIntersection.materialId];
				//float emit = shadowMaterial.emittance;

				if (shadowIntersection.t > 0.f)
				{
					if (shadowMaterial.emittance>0.f)
					{
						pathSegments[idx].color = materialIsc.color*shadowMaterial.color;
					}
					//inside shadow 
					else
					{
						pathSegments[idx].color *= glm::vec3(0.f);
					}
				}
				else
				{
					
					pathSegments[idx].color = glm::vec3(0.f);
				}
			}
		}
		else
		{
			pathSegments[idx].color = glm::vec3(0.f);
					
		}	
	}	
}

__global__ void MaterialKey(ShadeableIntersection* intersections, int* materialIdKey, int pixelCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < pixelCount)
	{
		materialIdKey[idx] = intersections[idx].materialId;
	}

}
//__global__ void StreamLeft(bool* steamFlag, PathSegment* pathsegments, int processLeft)
//{
//	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//	if (index < processLeft)
//	{
//		PathSegment pathSeg = pathsegments[index];
//		if (pathSeg.remainingBounces <= 0)
//		{
//			steamFlag[index] = false;
//		}
//		else
//		{
//			steamFlag[index] = true;
//		}
//	}
//}

//__global__ void TerminateColor(PathSegment* pathSegments, PathSegment* finalSegment,int pixelCount)
//{
//	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//	if (index < pixelCount)
//	{
//		if (pathSegments[index].remainingBounces == 0)
//		{
//			finalSegment[index] = pathSegments[index];
//		}
//	}
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
	const int blocksPerGrid1d = (pixelcount + blockSize1d - 1) / blockSize1d;


	//TODO steam compaction new arrays 
	int* steamCompactionOut;
	steamCompactionOut = (int*)malloc(sizeof(int)*pixelcount);
	int *newEnd = NULL;
	
	int processLeft = pixelcount;

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
    //     You may use either your implementation or `thrust`::remove_if` or its
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
	int compactSize = 0;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  int traceSize = num_paths;

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
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();
	depth++;

	

	//TODO material compactio
	MaterialKey << <blocksPerGrid1d, blockSize1d >> >(dev_intersections, materialKey, num_paths);
	thrust::device_ptr<int> dev_thrust_keys(materialKey);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_valuesInt(dev_intersections);
	thrust::device_ptr<PathSegment> dev_thrust_valueSeg(dev_paths);
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + pixelcount, dev_thrust_valuesInt);
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + pixelcount, dev_thrust_valueSeg);

	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

  //shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
  //  iter,
  //  num_paths,
  //  dev_intersections,
  //  dev_paths,
  //  dev_materials
  //);

	//**************************************************Naive Integrator**********************************************
	//TODO
	//Naive path tracing integrator
	//shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
	//	iter,
	//	num_paths,
	//	dev_intersections,
	//	dev_paths,
	//	dev_materials,
	//	traceDepth);

 //   BouncesLeft << <blocksPerGrid1d, blockSize1d >> > (dev_paths, compactSteamsIn, processLeft);
 //   cudaMemcpy(steamCompactionOut, compactSteamsIn, sizeof(int)*pixelcount, cudaMemcpyDeviceToHost);

 //   newEnd = thrust::remove(steamCompactionOut, steamCompactionOut + num_paths, 0);
 // 
 //   traceSize = newEnd - steamCompactionOut;

 //   if (newEnd == steamCompactionOut)
 //   {
 //   	  iterationComplete = true; // TODO: should be based off stream compaction results.
 //   }

	//**************************************End of naive integrator *****************************************************

	//**************************************Direct lighting***************************************************************
  //TODO
  //direct lighting path tracing integrator 
	DirectLightingIntegrator << <numblocksPathSegmentTracing, blockSize1d >> >
		(iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_geoms,
			1,
			traceDepth,
			hst_scene->geoms.size());

	iterationComplete = true;
	//****************************************End of direct lighting 
  
	}

	free(steamCompactionOut);

  // Assemble this iteration and apply it to the image
    
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
