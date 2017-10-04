#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <stb_image_write.h>
#include <stb_image.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/component_wise.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define CACHE_FIRST 0

#define MIS 1

#define ANTIALIAS 1

#define DOF 1
#define LENS_RADIUS 3.f
#define FOCAL_DIST  12.f

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



//Predicate functor
struct hasMoreBounces
{
	__host__ __device__
		bool operator()(const PathSegment& path)
	{
		return (path.remainingBounces > 0);
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
static ShadeableIntersection * dev_fst_bounce = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static unsigned char* dev_environment = NULL;

/*********************************************
**********************************************
******        Environment Mapping       ******
******             Functions            ******
**********************************************
**********************************************/

__device__ glm::vec3 getEnvMapColor(unsigned char* dev_environment, const glm::vec3& dir, const int& width, const int& height, const int& bpp) {
	float x = dir.x, y = dir.y, z = dir.z;

	float u = atan2f(x, z) / (2 * PI) + 0.5f;
	float v = y * 0.5f + 0.5f;

	v = 1-v;

	u *= width;
	v *= height;

	int u_i = u;
	int v_i = v;

	// Transform coordinates
	unsigned char* r = dev_environment + bpp * (u_i + width*v_i);

    glm::vec3 color = glm::vec3(*(r + 0), *(r + 1), *(r + 2));

	return glm::vec3(color.x, color.y, color.z) / 255.f;
}


void loadInEnvironment(const unsigned char* environment, const glm::ivec3 environment_dims) {
	int width  = environment_dims.x;
	int height = environment_dims.y;
	int bpp    = environment_dims.z;

	// Allocate CUDA array in device memory
	cudaMalloc(&dev_environment, bpp * width * height * sizeof(unsigned char));
	checkCUDAError("cudaMallocArray while mallocing texture array");

	cudaMemcpy(dev_environment, environment, bpp * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMempcpying while mallocing texture array");
}


void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	//Transfers Triangles from CPU to GPU

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	checkCUDAError("copying dev_geoms");

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_fst_bounce, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_fst_bounce, 0, pixelcount * sizeof(ShadeableIntersection));
	checkCUDAError("pathtraceInit");

	//Load In Environment Map
	if (scene->environment != NULL) {
		loadInEnvironment(scene->environment, scene->environment_dims);
		checkCUDAError("environmentLoading");
	}
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_fst_bounce);

	if (dev_environment != NULL) {
		cudaFree(dev_environment);
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

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
#if MIS
		segment.color = glm::vec3(0.f);
#else 
		segment.color = glm::vec3(1.f);
#endif
		segment.throughput = glm::vec3(1.f);

#if ANTIALIAS
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
		float jitter_x = u01(rng);
		float jitter_y = u01(rng);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter_x)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter_y)
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

__host__ __device__ void getIntersection(
	const Ray& ray
	, Geom* geoms
	, const int geoms_size
	, ShadeableIntersection& intersection) {

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
		const Geom& geom = geoms[i];

		//printf("Getting mesh intersection for: a %d type geom\n", geoms[i].type);

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == PLANE) {
			t = planeIntersectionTest(geom, ray, tmp_intersect, tmp_normal);
		}
		else if (geom.type == TRIANGLE) {
			t = triangleIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
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
		intersection.point = intersect_point;
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

		getIntersection(pathSegment.ray, geoms, geoms_size, intersections[path_index]);
	}
}

// A Naiive Integrator
__global__ void shadeMaterialNaive(
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
		PathSegment& path = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material m = materials[intersection.materialId];

			// If the material indicates that the object was a light, "light" the ray
			float pdf;
			scatterRay(path, intersection, m, rng, pdf);

		}
		else { // If there was no intersection, color the ray black.
			path.color = glm::vec3(0.0f);
			path.remainingBounces = 0;
		}
	}
}

// A Direct Lighting Integrator
__global__ void shadeMaterialDirect(
	int iter
	, int depth, int depthLimit
	, int light_count
	, int geoms_size
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, Geom* geoms
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& path = pathSegments[idx];
		if (intersection.t <= 0.0f) {
			path.color = glm::vec3(0.0f);
			path.remainingBounces = 0;
			return;
		}

		// if the intersection exists...
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material isect_m = materials[intersection.materialId];
		const glm::vec3& wo = -path.ray.direction;


		if (depth == 0 || path.specularBounce) {
			//405: Assumption: light is emitted equally from/to all directions
			glm::vec3 Le = isect_m.color * isect_m.emittance;
			path.color += path.throughput * Le;
		}

		if (isect_m.emittance > 0.f) {
			path.color = isect_m.color * isect_m.emittance;
			path.remainingBounces = 0;
			return;
		}

		path.specularBounce = isSpecular(isect_m.bsdf);

		if (!path.specularBounce) {
			/***************************************
			****************************************
			****************************************
			******* Light Importance Sampling ******
			****************************************
			****************************************
			****************************************/
			//Get f(X) and L(X)
			glm::vec3 wi;
			float pdf_li = 1.f;
			thrust::uniform_real_distribution<float> u02(0, light_count);
			int rand_li = u02(rng);

			const Geom light = geoms[rand_li];
			const Material light_m = materials[rand_li];
			glm::vec3 li_x = sample_li(light, light_m, intersection.point, rng, &wi, &pdf_li);  //Assuming lights give equal light from anywhere
			if (pdf_li < ZeroEpsilon) {
				// This is the shadow feeling part of my CPU Code:
				// Lines 71-81
				Ray dir_light;
				dir_light.origin = intersection.point + intersection.surfaceNormal * EPSILON;
				dir_light.direction = wi;
				ShadeableIntersection shadow_isect;
				getIntersection(dir_light, geoms, geoms_size, shadow_isect);

				// zero out contribution if it doesn't hit anything
				bool shadowed = shadow_isect.t > 0.f && (shadow_isect.materialId != light.materialid);
				li_x = shadowed ? glm::vec3(0) : li_x;

				const float pdf_bsdf = pdf(isect_m.bsdf, wo, -wi, intersection.surfaceNormal);

				////This only works because we have one bsdf in each material
				const glm::vec3 f_x = f(isect_m, wo, wi) * glm::abs(glm::dot(-wi, intersection.surfaceNormal));

				glm::vec3 Ld = (f_x * li_x)
					/ (pdf_li);

				Ld *= light_count;

				path.color += Ld;
				path.remainingBounces = 0;
				return;
			}
		}
	}
}


// A Broken Integrator
__global__ void shadeMaterialMIS(
	  int iter
	, int depth, int depthLimit
	, const int light_count
	, const int geoms_size
	, const int num_paths
	, unsigned char* dev_environment, const int map_width, const int map_height, const int map_bpp
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, Geom* geoms
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& path = pathSegments[idx];

		if (intersection.t < EPSILON) {
			//DO ENVIRONMENT MAPPING
			if (dev_environment != NULL) {
				path.color = getEnvMapColor(dev_environment, path.ray.direction, map_width, map_height, map_bpp);
			}
			else {
				path.color = glm::vec3(0.f);
			}

			path.remainingBounces = 0;
			return;
		}

		// if the intersection exists...
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material isect_m = materials[intersection.materialId];
		const glm::vec3& wo = -path.ray.direction;

		if (depth == 0 || path.specularBounce) {
			//405: Assumption: light is emitted equally from/to all directions
			glm::vec3 Le = isect_m.color * isect_m.emittance;
			path.color += path.throughput * Le;
		}

		if (isect_m.emittance > 0.f) {
			path.remainingBounces = 0;
			return;
		}

		//Store a copy. We only add this in the end.
		PathSegment gi_Component;
		gi_Component.color = glm::vec3(1.f);
		gi_Component.ray = path.ray;
		float gi_pdf;
		// If the material indicates that the object was a light, "light" the ray
		scatterRay(gi_Component, intersection, isect_m, rng, gi_pdf);

		__syncthreads();

		//Random Light Selection
		int rand_li = u01(rng)*light_count;
		const Geom light = geoms[rand_li];
		const Material light_m = materials[rand_li];

		//At this point, we've scattered, sampled and gi_component now has a new direction and origin.
		//            \      ^
		//             \    /
		//              \  /     <---- gi_Component
		//               \/
		//

		if (!path.specularBounce) {
			/***************************************
			****************************************
			****************************************
			******* Light Importance Sampling ******
			****************************************
			****************************************
			****************************************/
			glm::vec3 Ld = glm::vec3(0.f);

			//Get f(X) and L(X)
			glm::vec3 wi;
			float pdf_li = 1.f;

			glm::vec3 li_x = sample_li(light, light_m, intersection.point, rng, &wi, &pdf_li);  //Assuming lights give equal light from anywhere
			if (pdf_li > ZeroEpsilon) {
				// This is the shadow feeling part of my CPU Code:
				// Lines 71-81
				Ray dir_light_ray;
				dir_light_ray.origin = intersection.point + intersection.surfaceNormal * EPSILON;
				dir_light_ray.direction = wi;
				ShadeableIntersection shadow_isect;
				getIntersection(dir_light_ray, geoms, geoms_size, shadow_isect);

				// zero out contribution if it doesn't hit anything
				bool shadowed = shadow_isect.t > 0.f && (shadow_isect.materialId != light.materialid);
				li_x = shadowed ? glm::vec3(0) : li_x;

				const float pdf_bsdf = pdf(isect_m.bsdf, wo, -wi, intersection.surfaceNormal);

				////This only works because we have one bsdf in each material
				const glm::vec3 f_x = f(isect_m, wo, wi) * glm::abs(glm::dot(-wi, intersection.surfaceNormal));

				float weight_li = power_heuristic(1, pdf_li, 1, pdf_bsdf);

				Ld = (f_x * li_x * weight_li * path.throughput)
					/ (pdf_li);
			}//END DIRECT LIGHTING

			 /***************************************
			 ****************************************
			 ****************************************
			 ******* BSDF Importance Sampling *******
			 ****************************************
			 ****************************************
			 ****************************************/

			 //Get f(X) and L(X)
			PathSegment indirect_path;
			float bsdf_pdf;
			indirect_path.ray = path.ray;
			indirect_path.color = glm::vec3(1.f);
			scatterRay(indirect_path, intersection, isect_m, rng, bsdf_pdf);
			wi = indirect_path.ray.direction;
			glm::vec3 f_y = indirect_path.color;

			//Only do calculations if bsdfpdf is not zero for efficiency
			if (bsdf_pdf > ZeroEpsilon) {
				ShadeableIntersection bsdf_direct_isx;
				indirect_path.ray.origin = intersection.point + intersection.surfaceNormal * EPSILON;
				getIntersection(indirect_path.ray, geoms, geoms_size, bsdf_direct_isx);

				pdf_li = pdfLi(light, intersection, wi);

				//Only add cotribution if object hit is the light
				if (bsdf_direct_isx.t > 0 && bsdf_direct_isx.materialId == light.materialid) {
					float weight_bsdf = power_heuristic(1, bsdf_pdf, 1, pdf_li);

					glm::vec3 li_y = light_m.emittance * light_m.color;

					Ld += li_y * f_y * weight_bsdf * path.throughput;
				}
			}

			Ld *= light_count;

			//****************************************
			//**Add Ld to Ray color before GI Stuff***
			//****************************************
			path.color += Ld;
		}

		//Update Scene_Ray - This just spawns a new ray for the next loop
		path.ray = gi_Component.ray;
		path.throughput *= gi_Component.color;
		path.specularBounce = isSpecular(isect_m.bsdf);

		if ((isBlack(gi_Component.color)) || intersection.materialId == light.materialid) {
			path.remainingBounces = 0;
			return;
		}

		//Russian Roulette Early Ray Termination
		if (depth >= 3) {
			float q = glm::max(0.05f, (1 - glm::compMax(path.throughput)));
			if (u01(rng) < q) {
				path.remainingBounces = 0;
				return;
			}
			path.throughput /= (1 - q);
		}
	}
}

// A Backup/Just in Case Integrator
// Who needs source control amirite
__global__ void shadeMaterialMIS_backup(
	int iter
	, int depth, int depthLimit
	, int light_count
	, int geoms_size
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, Geom* geoms
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& path = pathSegments[idx];
		if (intersection.t <= 0.0f) {
			path.color = glm::vec3(0.0f);
			path.remainingBounces = 0;
			return;
		}

		// if the intersection exists...
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material isect_m = materials[intersection.materialId];
		const glm::vec3& wo = -path.ray.direction;


		if (depth == 0 || path.specularBounce) {
			//405: Assumption: light is emitted equally from/to all directions
			glm::vec3 Le = isect_m.color * isect_m.emittance;
			path.color += path.throughput * Le;
		}

		if (isect_m.emittance > 0.f) {
			path.color = isect_m.color * isect_m.emittance;
			path.remainingBounces = 0;
			return;
		}

		//Store a copy. We only add this in the end.
		PathSegment gi_Component = PathSegment(path);
		gi_Component.color = glm::vec3(1.f);
		float gi_pdf;

		// If the material indicates that the object was a light, "light" the ray
		scatterRay(gi_Component, intersection, isect_m, rng, gi_pdf);

		path.specularBounce = isect_m.bsdf == 1 || isect_m.bsdf == 2;

		thrust::uniform_real_distribution<float> u02(0, light_count);
		int rand_li = u02(rng);
		const Geom light = geoms[rand_li];
		const Material light_m = materials[rand_li];


		//At this point, we've scattered, sampled and gi_component now has a new direction and origin.
		//            \      ^
		//             \    /
		//              \  /     <---- gi_Component
		//               \/
		//
		if (!path.specularBounce) {
			/***************************************
			****************************************
			****************************************
			******* Light Importance Sampling ******
			****************************************
			****************************************
			****************************************/
			//Get f(X) and L(X)
			glm::vec3 wi;
			float pdf_li = 1.f;
			glm::vec3 li_x = sample_li(light, light_m, intersection.point, rng, &wi, &pdf_li);  //Assuming lights give equal light from anywhere
			if (pdf_li != 0.f) {
				// This is the shadow feeling part of my CPU Code:
				// Lines 71-81
				Ray dir_light;
				dir_light.origin = intersection.point + intersection.surfaceNormal * EPSILON;
				dir_light.direction = wi;
				ShadeableIntersection shadow_isect;
				getIntersection(dir_light, geoms, geoms_size, shadow_isect);

				// zero out contribution if it doesn't hit anything
				bool shadowed = shadow_isect.t > 0.f && (shadow_isect.materialId != light.materialid);
				li_x = shadowed ? glm::vec3(0) : li_x;

				const float pdf_bsdf = pdf(isect_m.bsdf, wo, -wi, intersection.surfaceNormal);

				////This only works because we have one bsdf in each material
				const glm::vec3 f_x = f(isect_m, wo, wi) * glm::abs(glm::dot(-wi, intersection.surfaceNormal));

			    float weight_li = power_heuristic(1, pdf_li, 1, pdf_bsdf);

				glm::vec3 Ld = (f_x * li_x * weight_li) 
					                / (pdf_li);

				//Ld *= path.throughput;
				//DELET THIS
				Ld *= light_count;

				path.color += Ld;
				//DELET THIS
				//path.color = Ld;
				path.remainingBounces = 0;
				return;
			}

			if (isBlack(gi_Component.color) || intersection.materialId == light.materialid || gi_pdf == 0.f) {
				path.remainingBounces = 0;

				if (isBlack(gi_Component.color)) printf("IS HOMIE HOMIE \n");
				if (intersection.materialId == light.materialid) printf("CASE 2: MAT THING \n");
				if (gi_pdf == 0) printf("GI_PDF IS ZERO HOMIE \n");
				return;
			}

		}

		//Update Scene_Ray - This just spawns a new ray for the next loop
		path.ray = gi_Component.ray;
		path.throughput *= gi_Component.color;
		path.remainingBounces--;
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
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete && depth < traceDepth) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST
		if ((depth == 0 && iter == 1) || depth > 0) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();

			if (depth == 0) {
				cudaMemcpy(dev_fst_bounce, dev_intersections, sizeof(ShadeableIntersection) * num_paths, cudaMemcpyDeviceToDevice);
			}
		}
		else if (depth == 0) {
			cudaMemcpy(dev_intersections, dev_fst_bounce, sizeof(ShadeableIntersection) * num_paths, cudaMemcpyDeviceToDevice);
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
		checkCUDAError("computed intersections");
		cudaDeviceSynchronize();
#endif

		// --- Shading Stage ---
#if MIS
		int lc = hst_scene->light_count;
		shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth, hst_scene->state.traceDepth,
			lc,
			hst_scene->geoms.size(),
			num_paths,
			dev_environment, hst_scene->environment_dims[0], hst_scene->environment_dims[1], hst_scene->environment_dims[2],
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_geoms
			);
		checkCUDAError("shadeMaterialMIS");
#else
		shadeMaterialNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
#endif

		cudaDeviceSynchronize();

		// --- Stream Compaction
		PathSegment* remaining_end =
			thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, hasMoreBounces());
		num_paths = remaining_end - dev_paths;

		iterationComplete = num_paths == 0;
		depth++;
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	num_paths = dev_path_end - dev_paths;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
