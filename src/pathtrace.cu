#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>


#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "stream_compaction/EfficientStreamCompaction.h"

#include "bounds.h"

#define ERRORCHECK 1
// ----------------------------------------------------------------
//----------------------- Toggle Here -----------------------------
// ----------------------------------------------------------------

// Uncommment to disable First bounce ray and intersections cache
//#define NO_FIRSTBOUNCECACHE

// Uncommment to enable Stochastic Sampled Antialiasing
#define AA_STOCHASTIC

// Uncommment to enable sorting by material ID
//#define THRUSTSORT
//#define RADIXSORT

// Uncommment to enable len persoective camera
//#define LENPERSPECTIVECAMERA
//#define LEN_RADIUS 3.0f
//#define FOCAL_LENGTH 13.0f
// ----------------------------------------------------------------
// ----------------------------------------------------------------

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

// TODO: static variables for device memory, any extra info you need, etc

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;

static PathSegment * dev_paths = NULL;
static PathSegment * dev_first_bounce_paths = NULL;

static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_first_bounce_intersections = NULL;

static int NoInteresctMaterialID;

// Radix Sort Stuff
static int* radix_sort_e_array;
static int* radix_sort_f_array;
static int* radix_sort_totalFalses;

static RadixSortElement* dev_RadixSort = NULL;
static RadixSortElement* dev_RadixSort_after_sort = NULL;

static PathSegment * dev_paths_after_sort = NULL;
static ShadeableIntersection * dev_intersections_after_sort = NULL;

// mesh part
static Triangle * dev_tris = NULL;

static int worldBoundsSize = 0;
static int bvhNodesSize = 0;


// texture part
static Texture * dev_textureMap = NULL;
static int textureMapSize = 0;

static Texture * dev_normalMap = NULL;
static int normalMapSize = 0;

static Texture * dev_environmentMap = NULL;


#ifdef  ENABLE_DIR_LIGHTING
static Light * dev_lights = NULL;
#endif

static int lightSize = 0;

#ifdef ENABLE_MESHWORLDBOUND
static Bounds3f * dev_worldBounds = NULL;
#endif 
#ifdef ENABLE_BVH
static LinearBVHNode* dev_bvh_nodes = NULL;
#endif

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));


  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
	
	// Triangles of mesh
	cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
#ifdef ENABLE_MESHWORLDBOUND
	worldBoundsSize = scene->worldBounds.size();
	if (worldBoundsSize > 0) {
		// World bounds of mesh
		cudaMalloc(&dev_worldBounds, worldBoundsSize * sizeof(Bounds3f));
		cudaMemcpy(dev_worldBounds, scene->worldBounds.data(), worldBoundsSize * sizeof(Bounds3f), cudaMemcpyHostToDevice);
	}
#endif
#ifdef ENABLE_BVH
	bvhNodesSize = scene->bvh_totalNodes;
	if (bvhNodesSize > 0) {
		//BVH Nodes
		cudaMalloc(&dev_bvh_nodes, bvhNodesSize * sizeof(LinearBVHNode));
		cudaMemcpy(dev_bvh_nodes, scene->bvh_nodes, bvhNodesSize * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
	}
#endif



  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_bounce_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

	// Radix Sort Stuff
	cudaMalloc(&dev_paths_after_sort, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_intersections_after_sort, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_after_sort, 0, pixelcount * sizeof(ShadeableIntersection));

	//cudaMalloc((void**)&radix_sort_b_array, pixelcount * sizeof(int));
	cudaMalloc((void**)&radix_sort_e_array, pixelcount * sizeof(int));
	cudaMalloc((void**)&radix_sort_f_array, pixelcount * sizeof(int));

	cudaMalloc((void**)&radix_sort_totalFalses, sizeof(int));

	cudaMalloc((void**)&dev_RadixSort, pixelcount * sizeof(RadixSortElement));
	cudaMalloc((void**)&dev_RadixSort_after_sort, pixelcount * sizeof(RadixSortElement));

	NoInteresctMaterialID = hst_scene->materials.size();

#ifdef  ENABLE_DIR_LIGHTING
	lightSize = scene->lights.size();
	if (lightSize > 0) {
		//BVH Nodes
		cudaMalloc(&dev_lights, lightSize * sizeof(Light));
		cudaMemcpy(dev_lights, scene->lights.data(), lightSize * sizeof(Light), cudaMemcpyHostToDevice);
	}
#endif

	// Texture -> GPU side
	textureMapSize = scene->textureMap.size();
	if (textureMapSize > 0) {
		cudaMalloc(&dev_textureMap, textureMapSize * sizeof(Texture));
		for (int i = 0; i < textureMapSize; i++) {
			int num_of_elements = scene->textureMap[i].width * scene->textureMap[i].height * scene->textureMap[i].n_comp;
			// assgin new GPU side data ptr
			cudaMalloc(&(scene->textureMap[i].dev_data), num_of_elements * sizeof(unsigned char));
			// move texture data first
			cudaMemcpy(scene->textureMap[i].dev_data, scene->textureMap[i].host_data, num_of_elements * sizeof(unsigned char), cudaMemcpyHostToDevice);
		}
		// move whole texture
		cudaMemcpy(dev_textureMap, scene->textureMap.data(), textureMapSize * sizeof(Texture), cudaMemcpyHostToDevice);
	}

	// noraml Map -> GPU side
	normalMapSize = scene->normalMap.size();
	if (normalMapSize > 0) {
		cudaMalloc(&dev_normalMap, normalMapSize * sizeof(Texture));
		for (int i = 0; i < normalMapSize; i++) {
			int num_of_elements = scene->normalMap[i].width * scene->normalMap[i].height * scene->normalMap[i].n_comp;
			// assgin new GPU side data ptr
			cudaMalloc(&(scene->normalMap[i].dev_data), num_of_elements * sizeof(unsigned char));
			// move texture data first
			cudaMemcpy(scene->normalMap[i].dev_data, scene->normalMap[i].host_data, num_of_elements * sizeof(unsigned char), cudaMemcpyHostToDevice);
		}
		// move whole texture
		cudaMemcpy(dev_normalMap, scene->normalMap.data(), normalMapSize * sizeof(Texture), cudaMemcpyHostToDevice);
	}

	// environment Map -> GPU side
	// ONLY ONE environment map or nothing
	int environmentMapSize = scene->EnvironmentMap.size();
	if (environmentMapSize > 0) {
		cudaMalloc(&dev_environmentMap, sizeof(Texture));
		
		int num_of_elements = scene->EnvironmentMap[0].width * scene->EnvironmentMap[0].height * scene->EnvironmentMap[0].n_comp;
		// assgin new GPU side data ptr
		cudaMalloc(&(scene->EnvironmentMap[0].dev_data), num_of_elements * sizeof(unsigned char));
		// move texture data first
		cudaMemcpy(scene->EnvironmentMap[0].dev_data, scene->EnvironmentMap[0].host_data, num_of_elements * sizeof(unsigned char), cudaMemcpyHostToDevice);
		
		// move whole texture
		cudaMemcpy(dev_environmentMap, scene->EnvironmentMap.data(), sizeof(Texture), cudaMemcpyHostToDevice);
	}



	cudaDeviceSynchronize();

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null

  	cudaFree(dev_paths);
	cudaFree(dev_first_bounce_paths);

  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);

  	cudaFree(dev_intersections);
	cudaFree(dev_first_bounce_intersections);

    // TODO: clean up any extra device memory you created

	// Radix Sort Stuff
	cudaFree(dev_paths_after_sort);
	cudaFree(dev_intersections_after_sort);

	//cudaFree(radix_sort_b_array);
	cudaFree(radix_sort_e_array);
	cudaFree(radix_sort_f_array);

	cudaFree(radix_sort_totalFalses);
	cudaFree(dev_RadixSort);
	cudaFree(dev_RadixSort_after_sort);

	cudaFree(dev_tris);

#ifdef ENABLE_MESHWORLDBOUND
	if (worldBoundsSize > 0) {
		cudaFree(dev_worldBounds);
	}
#endif
#ifdef ENABLE_BVH
	if (bvhNodesSize > 0) {
		cudaFree(dev_bvh_nodes);
	}
#endif

#ifdef  ENABLE_DIR_LIGHTING
	if (lightSize > 0) {
		cudaFree(dev_lights);
	}
#endif

	if (textureMapSize > 0) {
		// TODO : Unknown error here, try to fix it
		// Or is this the right way?
		//for (int i = 0; i < textureMapSize; i++) {
		//	cudaFree(dev_textureMap[i].dev_data);
		//}
		//cudaDeviceSynchronize();
		cudaFree(dev_textureMap);
	}

	if (normalMapSize > 0) {
		//...
		cudaFree(dev_normalMap);
	}

	if (dev_environmentMap != NULL) {
		//...
		cudaFree(dev_environmentMap);
	}

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


__host__ __device__
glm::vec2 squareToDiskConcentric(const float x, const float y) {
	float r = x;
	float theta = 2.0f * PI * y;
	return glm::vec2(r * cos(theta), r * sin(theta));
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);


#if defined(AA_STOCHASTIC) || defined(LENPERSPECTIVECAMERA)
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		float x_offset = u01(rng);
		float y_offset = u01(rng);
#endif 

		PathSegment& segment = pathSegments[index];
		glm::vec3 pos = cam.position;
		glm::vec3 dir;

#ifdef ENABLE_DIR_LIGHTING
		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);

#ifdef ENABLE_MIS_LIGHTING
		segment.ThroughputColor = glm::vec3(1.0f, 1.0f, 1.0f);
#endif 

#else
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#endif 

		


#ifdef AA_STOCHASTIC
		// implement antialiasing by jittering the ray
		// We regard every ray has the same weight so far

		 dir = glm::normalize(cam.view
			 - cam.right * cam.pixelLength.x * ((float)x + x_offset - (float)cam.resolution.x * 0.5f)
			 - cam.up * cam.pixelLength.y * ((float)y + y_offset - (float)cam.resolution.y * 0.5f)
		     );
#else
		dir = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif	


#ifdef LENPERSPECTIVECAMERA
		glm::vec2 pLens = LEN_RADIUS * squareToDiskConcentric(x_offset, y_offset);
		
		// Distance to focal plane
		float ft = FOCAL_LENGTH / AbsDot(cam.view, dir);
		glm::vec3 pFocus = pos + ft * dir;

		pos += pLens.y * cam.up;
		pos += pLens.x * cam.right;

		dir = glm::normalize(pFocus - pos);
#endif 


		segment.ray.origin = pos;
		segment.ray.direction = dir;

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, Triangle * tris
	, int geoms_size
	, ShadeableIntersection * intersections
	, int noInteresctMaterialID
#ifdef ENABLE_MESHWORLDBOUND
	, Bounds3f * worldBounds
#endif
#ifdef ENABLE_BVH
	, LinearBVHNode * nodes
#endif
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		//glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		//glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_uv, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_uv, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH)
			{	
#ifdef ENABLE_MESHWORLDBOUND
#ifdef ENABLE_BVH
				ShadeableIntersection temp_isect;
				temp_isect.t = FLT_MAX;
				if (IntersectBVH(pathSegment.ray, &temp_isect, nodes, tris)) {
					t = temp_isect.t;
					tmp_uv = temp_isect.uv;
					tmp_normal = temp_isect.surfaceNormal;
				}
				else {
					t = -1.0f;
				}
#else
				// Check geom related world bound first
				float tmp_t;
				if (worldBounds[geom.worldBoundIdx].Intersect(pathSegment.ray, &tmp_t)) {
					t = meshIntersectionTest(geom, tris, pathSegment.ray, tmp_uv, tmp_normal, outside);
				}
				else {
					t = -1.0f;
				}
#endif
#else
				// loop through all triangles in related mesh
				t = meshIntersectionTest(geom, tris, pathSegment.ray, tmp_uv, tmp_normal, outside);
#endif		
			}


			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				//intersect_point = tmp_intersect;
				uv = tmp_uv;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			// we need sort by this, assign -1 for those have no intersections
			intersections[path_index].materialId = noInteresctMaterialID; // May need to change this number
		}
		else
		{	
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].uv = uv;
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
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
		glm::vec3 debug_Color = (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= debug_Color;
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

// First Version of Shade function
__global__ void shadeMaterialNaive(
	  int iter
	, int num_paths
	, int depth 
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, Texture * textures
	, Texture * normalMaps
	, Texture * environmentMap
#ifdef ENABLE_DIR_LIGHTING
	, Light * lights
	, int lightSize
	, Geom * geoms
	, Triangle * tris
	, int geomSize
#ifdef ENABLE_MESHWORLDBOUND
	, Bounds3f * worldBounds
#endif
#ifdef ENABLE_BVH
	, LinearBVHNode * nodes
#endif
#endif 
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& pathSegment = pathSegments[idx];

		if (pathSegment.remainingBounces == 0) return;

		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;


			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
#ifdef	ENABLE_DIR_LIGHTING

#ifdef ENABLE_MIS_LIGHTING
				pathSegment.color += (pathSegment.ThroughputColor * (materialColor * material.emittance));
#else
				pathSegment.color += (materialColor * material.emittance);
#endif

#else
				pathSegment.color *= (materialColor * material.emittance);
#endif
				// Terminate the ray if it hist a light
				pathSegment.remainingBounces = 0;
			}
			
			else {
				glm::vec3 isect_normal = intersection.surfaceNormal;

				if (material.normalID != -1) {
					isect_normal = normalMaps[material.normalID].getNormal(intersection.uv);
				}

				scatterRay(pathSegment, 
						   getPointOnRay(pathSegment.ray, intersection.t),
						   isect_normal,
						   intersection.uv,
					       material,
						   rng,
						   textures
#ifdef	ENABLE_DIR_LIGHTING
						  , lights
						  , lightSize
						  , geoms
						  , tris
						  , geomSize
#ifdef ENABLE_MESHWORLDBOUND
						  , worldBounds
#endif
#ifdef ENABLE_BVH
						  , nodes
#endif
#endif				
						);

				pathSegment.remainingBounces--;
			}	
		}
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
		else {
			pathSegment.color = glm::vec3(0.0f);

			if (environmentMap != NULL) {
				pathSegment.color = environmentMap[0].getEnvironmentColor(pathSegment.ray.direction);
			}

			pathSegment.remainingBounces = 0;
		}
	}
}




// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths, glm::vec3 ambientColor)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += (iterationPath.color + ambientColor);
	}
}


// ----------------- Radix Sort ------------------------
__global__ void kernGen_e_array(int N, int idxBit, int* e_array, const RadixSortElement *dev_data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	int temp_result = (dev_data[index].materialId >> idxBit) & 1;
	e_array[index] = 1 - temp_result;
}

template<int SIZE>
__global__ void kernGen_d_array_and_scatter(int N, const int* dev_totalFalse, const int* e_array, const int* f_array, 
	RadixSortElement* dev_odata, const RadixSortElement* dev_idata)
	//ShadeableIntersection* dev_odata1, const ShadeableIntersection* dev_idata1,
	//PathSegment* dev_odata2, const PathSegment* dev_idata2
	//)
{	
	__shared__ int totalFalse[SIZE];

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	// Latency hiding
	//ShadeableIntersection temp1 = dev_idata1[index];
	//PathSegment temp2 = dev_idata2[index];

	RadixSortElement temp = dev_idata[index];

	if (threadIdx.x == 0) {
		totalFalse[0] = dev_totalFalse[0];
	}
	__syncthreads();


	int t_array_value = index - f_array[index] + totalFalse[0];

	//int d_array_value = b_array[index] ? t_array_value : f_array[index];
	int d_array_value = e_array[index] ?  f_array[index] : t_array_value;

	//dev_odata1[d_array_value] = dev_idata1[index];
	//dev_odata2[d_array_value] = dev_idata2[index];

	//dev_odata1[d_array_value] = temp1;
	//dev_odata2[d_array_value] = temp2;

	dev_odata[d_array_value] = temp;
}

__global__ void kernGetTotalFalses(int N, int* totalFalses, const int* e_array, const int* f_array) {
	totalFalses[0] = e_array[N - 1] + f_array[N - 1];
}


__global__ void kernGenRadixSortElementArray(int N, RadixSortElement* dev_RadixSort, ShadeableIntersection* dev_intersections) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	ShadeableIntersection temp = dev_intersections[index];

	dev_RadixSort[index].OriIndex = index;
	dev_RadixSort[index].materialId = temp.materialId;
}

__global__ void kernSortByRadixSortResult(int N, const RadixSortElement* dev_RadixSort,
	ShadeableIntersection* dev_odata1, const ShadeableIntersection* dev_idata1,
	PathSegment* dev_odata2, const PathSegment* dev_idata2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	int OriIndex = dev_RadixSort[index].OriIndex;
	dev_odata1[index] = dev_idata1[OriIndex];
	dev_odata2[index] = dev_idata2[OriIndex];
}
// ------------------------------------------------------

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

    // Perform one iteration of path tracing
	int depth = 0;
	//PathSegment* dev_path_end = dev_paths + pixelcount;
	//int num_paths = dev_path_end - dev_paths;
	int num_paths = pixelcount;


#if defined(AA_STOCHASTIC) || defined(NO_FIRSTBOUNCECACHE)
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	computeIntersections << <dim3((pixelcount + blockSize1d - 1) / blockSize1d), blockSize1d >> > (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, dev_tris
		, hst_scene->geoms.size()
		, dev_intersections
		, NoInteresctMaterialID
#ifdef ENABLE_MESHWORLDBOUND
		, dev_worldBounds
#endif
#ifdef ENABLE_BVH
		, dev_bvh_nodes
#endif
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();

#else

	// Cache camera rays and the intersections of these rays 
	// the very first sample per pixel
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_first_bounce_paths);
		checkCUDAError("generate camera ray");

		// clean shading chunks
		cudaMemset(dev_first_bounce_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		computeIntersections << <dim3((pixelcount + blockSize1d - 1) / blockSize1d), blockSize1d >> > (
			depth
			, num_paths
			, dev_first_bounce_paths
			, dev_geoms
			, dev_tris
			, hst_scene->geoms.size()
			, dev_first_bounce_intersections
			, NoInteresctMaterialID
#ifdef ENABLE_MESHWORLDBOUND
			, dev_worldBounds
#endif
#ifdef ENABLE_BVH
			, dev_bvh_nodes
#endif
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
	}

	// Directly copy first camera rays and their intersections from the very first sample per pixel
	cudaMemcpy(dev_paths, dev_first_bounce_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_intersections, dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
#endif


	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
	while (!iterationComplete) {

		// tracing
		// num_paths : number of paths/pathSegments that we should still trace
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		// Do not compute intersections when depth is 0,
		// Already copy it from dev_first_bounce_intersections
		if (depth) {
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, dev_tris
				, hst_scene->geoms.size()
				, dev_intersections
				, NoInteresctMaterialID
#ifdef ENABLE_MESHWORLDBOUND
				, dev_worldBounds
#endif
#ifdef ENABLE_BVH
				, dev_bvh_nodes
#endif
				);
			checkCUDAError("trace one bounce");

			cudaDeviceSynchronize();
		}

		depth++;
		
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	    // evaluating the BSDF.
	    // Start off with just a big kernel that handles all the different
	    // materials you have in the scenefile.

	    // TODO: compare between directly shading the path segments and shading
	    // path segments that have been reshuffled to be contiguous in memory.

		// Sort here by ShadeableIntersection's material ID

#ifdef THRUSTSORT
		// ------------------------ thrust Sort -------------------------
		thrust::device_ptr<ShadeableIntersection> dev_thrust_keys(dev_intersections);
		thrust::device_ptr<PathSegment> dev_thrust_values(dev_paths);
		thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + num_paths, dev_thrust_values);
#else
#ifdef RADIXSORT
		// ------------------------ Radix Sort -------------------------
		kernGenRadixSortElementArray << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_RadixSort, dev_intersections);

		// 7 kinds of materials, numbOfBits -> 3
		int numOfBits = ilog2ceil(hst_scene->materials.size());

		for (int k = 0; k <= numOfBits; k++) {
			// This should based on material ID
			kernGen_e_array << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, k, radix_sort_e_array, dev_RadixSort);

			// Get Exclusive scan result as a whole
			StreamCompaction::Efficient::scanDynamicShared(num_paths, radix_sort_f_array, radix_sort_e_array);

			kernGetTotalFalses << <dim3(1), dim3(1) >> > (num_paths, radix_sort_totalFalses, radix_sort_e_array, radix_sort_f_array);

			// Since here we run exclusive scan as a whole,
			// and we don't want each tile to run StreamCompaction::Efficient::scan individually.
			// value in d_array here is actually index value in the whole data array, not just index in that tile
			// so, there is NO need to merge here
			kernGen_d_array_and_scatter<1> << <numblocksPathSegmentTracing, blockSize1d >> >
				(num_paths, radix_sort_totalFalses,
				 radix_sort_e_array, radix_sort_f_array, 
				 dev_RadixSort_after_sort, dev_RadixSort);
				//dev_intersections_after_sort, dev_intersections,
				//dev_paths_after_sort, dev_paths);

			RadixSortElement* temp = dev_RadixSort;
			dev_RadixSort = dev_RadixSort_after_sort;
			dev_RadixSort_after_sort = temp;	

			//ShadeableIntersection* temp1 = dev_intersections;
			//dev_intersections = dev_intersections_after_sort;
			//dev_intersections_after_sort = temp1;

			//PathSegment* temp2 = dev_paths;
			//dev_paths = dev_paths_after_sort;
			//dev_paths_after_sort = temp2;
		}
		//cudaDeviceSynchronize();

		kernSortByRadixSortResult << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_RadixSort,
			dev_intersections_after_sort, dev_intersections,
			dev_paths_after_sort, dev_paths);

		cudaMemcpy(dev_paths, dev_paths_after_sort, sizeof(PathSegment) * num_paths, cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_intersections, dev_intersections_after_sort, sizeof(ShadeableIntersection) * num_paths, cudaMemcpyDeviceToDevice);
		// -----------------------------------------------------------------
#endif
#endif

	   // shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
		  //iter,
		  //num_paths,
		  //dev_intersections,
		  //dev_paths,
		  //dev_materials
	   // );

		shadeMaterialNaive<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			depth,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_textureMap,
			dev_normalMap,
			dev_environmentMap
#ifdef ENABLE_DIR_LIGHTING
			, dev_lights
			, lightSize
			, dev_geoms
			, dev_tris
			, hst_scene->geoms.size()
#ifdef ENABLE_MESHWORLDBOUND
			, dev_worldBounds
#endif
#ifdef ENABLE_BVH
			, dev_bvh_nodes
#endif
#endif 
		);
		checkCUDAError("shadeMaterial error");

		
		// actually, for the last path tracing, there is not need to 
		// conduct stream compaction, because there is no following path-tracing
		if (depth == traceDepth) {
			iterationComplete = true; 
			continue;
		}

		// num_paths : number of paths/pathSegments that we should still trace
		// iterationComplete should be based off stream compaction results
		num_paths = StreamCompaction::Efficient::compactDynamicShared(num_paths, dev_paths);

		// if there is no paths meet the requirement(remaining bounces > 0), we just end the loop
		if (num_paths == 0) {
			iterationComplete = true;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

	glm::vec3 ambientColor = glm::vec3(0.1f, 0.1f, 0.1f);

	finalGather << <numBlocksPixels, blockSize1d >> >(pixelcount, dev_image, dev_paths, ambientColor);
	checkCUDAError("final Gather");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
	checkCUDAError("sendImageToPBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
