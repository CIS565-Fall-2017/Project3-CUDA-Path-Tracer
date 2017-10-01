#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define DIRECTLIGHT 0
#define REALISTIC_CAMERA 0
#define MOVINGTRACE 0
#define CACHEFIRST 0
#define	SORTMATERIAL 0
#define TIMERCOUNT 0

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
static ShadeableIntersection * dev_cache_intersections =  NULL;
static int DirectLight_Count = 0;
//My directlight index
static int* dev_LightIndex = NULL;
static int pcount = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...
int *material_id = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
	//scene->state.traceDepth += 10;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
	pcount = pixelcount;
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_LightIndex, scene->geoms.size() * sizeof(int));
	cudaMalloc(&material_id, pixelcount * sizeof(int));
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_LightIndex);
	cudaFree(dev_cache_intersections);
	cudaFree(material_id);
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

//======================================================================================================
//======================================================================================================
struct AlivePathTest {
	__host__ __device__
		bool operator()(const PathSegment &path) {
		return path.remainingBounces <= 0;
	}
};
__global__ void shadeAllMaterial(int iter, int num_paths,
	ShadeableIntersection *shadeableIntersections,
	PathSegment *pathSegments, Material *materials) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_paths) {
		return;
	}

	ShadeableIntersection intersection = shadeableIntersections[idx];

	if (intersection.t > 0.0f) {

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		const Material &material = materials[intersection.materialId];

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.f) {
			pathSegments[idx].remainingBounces = 0;
			pathSegments[idx].color *= material.color * material.emittance;
		}
		else {
			pathSegments[idx].color *= material.color;
		}
	}
	else {
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = 0;
	}
	/*
	float colorRGB = pathSegments[idx].color.x + pathSegments[idx].color.y
		+ pathSegments[idx].color.z;

	if (colorRGB <= 0.f) {
		pathSegments[idx].remainingBounces = 0;
	}
	*/
}

// Add the current iteration's output of terminated path to the overall image
__global__ void gatherAndTerminate(int num_paths, glm::vec3 * image,
	PathSegment *dev_paths, ShadeableIntersection *dev_intersections) {

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= num_paths) {
		return;
	}

	PathSegment &path = dev_paths[idx];
	ShadeableIntersection &intersection = dev_intersections[idx];

	path.remainingBounces -= 1;
	if (path.remainingBounces <= 0) {
		image[path.pixelIndex] += path.color;
		dev_paths[idx].color = glm::vec3(0.f);
	}
	if (intersection.t <= 1e-4f) {
		path.remainingBounces = 0;
	}
}

__global__ void scatterAliveRays(int iter, int depth, int num_paths,
	PathSegment *dev_paths, ShadeableIntersection *dev_intersections,
	Material *dev_materials) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= num_paths) {
		return;
	}

	PathSegment &path = dev_paths[idx];
	ShadeableIntersection &intersection = dev_intersections[idx];
	Material &material = dev_materials[intersection.materialId];
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

	if (path.remainingBounces <= 0) {
		return;
	}

	scatterRay(path,path.ray.origin, intersection.surfaceNormal,intersection.t, material, rng);
}
__device__ float AbsDot(const glm::vec3 &a, const glm::vec3 &b)
{
	return glm::abs(glm::dot(a, b));
};
__device__ int fake_sampleLi(
	Geom* geoms,
	float &pdfLi,
	glm::vec4 &global_coor_light
	) {

}
__global__ void shadeDirectLight(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, int *dev_LightIndex
	, Geom *dev_geoms
	, int light_sum
	, int geoms_size
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
				pathSegments[idx].color = (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				glm::vec3 Intersection_Coordinate = (intersection.t) * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin + intersection.surfaceNormal * 0.0001f;
				//Find a light. Here we assume the light is a cube. And only 1 light
				thrust::uniform_real_distribution<float> uLight(0, light_sum);
				int Light_idx = dev_LightIndex[int(floor(uLight(rng)))];
				double lx = dev_geoms[Light_idx].scale[0];
				double ly = dev_geoms[Light_idx].scale[2];

				glm::vec3 LightColor = materials[dev_geoms[Light_idx].materialid].color *
					materials[dev_geoms[Light_idx].materialid].emittance;
				glm::mat4 light_trans = dev_geoms[Light_idx].transform;
				thrust::uniform_real_distribution<float> ux(-1.0f / 2.0, 1.0f / 2.0);
				thrust::uniform_real_distribution<float> uy(-1.0f / 2.0, 1.0f / 2.0);
				thrust::uniform_real_distribution<float> uz(-1.0f / 2.0, 1.0f / 2.0);
				int face_case = int(floor(u01(rng) * 6));
				glm::vec4 local_coor = glm::vec4(ux(rng), -0.5, uz(rng), 1);
				
				//switch (face_case) {
				//	case 0:
				//		local_coor = glm::vec4(0.5, uy(rng), uz(rng), 1);
				//		break;
				//	case 1:
				//		local_coor = glm::vec4(-0.5, uy(rng), uz(rng), 1);
				//		break;
				//	case 2:
				//		local_coor = glm::vec4(ux(rng), 0.5, uz(rng), 1);
				//		break;
				//	case 3:
				//		local_coor = glm::vec4(ux(rng), -0.5, uz(rng), 1);
				//		break;
				//	case 4:
				//		local_coor = glm::vec4(ux(rng), uy(rng), 0.5, 1);
				//		break;
				//	case 5:
				//		local_coor = glm::vec4(ux(rng), uy(rng), -0.5, 1);
				//		break;
				//	default:
				//		break;
				//}
				/*
				glm::vec4 local_coor = glm::vec4(ux(rng), -0.5, uy(rng), 1);
				*/
				glm::vec4 global_coor_light = light_trans * local_coor;
				float length = glm::length(glm::vec3(global_coor_light) - Intersection_Coordinate);
				float pdf_li = length * length / (
					dev_geoms[Light_idx].scale[0] * dev_geoms[Light_idx].scale[1] +
					dev_geoms[Light_idx].scale[0] * dev_geoms[Light_idx].scale[2] +
					dev_geoms[Light_idx].scale[1] * dev_geoms[Light_idx].scale[2]
					) / 2.0f;
				//Now we have pdf_li and position of point on light. we need to find if there is any shadow.
				//Shadow test
				float t;
				glm::vec3 intersect_point;
				glm::vec3 normal;
				float t_min = FLT_MAX;
				int hit_geom_index = -1;
				bool outside = true;
				Ray fromIsxToLight;
				glm::vec3 tmp_intersect;
				glm::vec3 tmp_normal;
				fromIsxToLight.origin = Intersection_Coordinate;
				fromIsxToLight.direction = glm::normalize(glm::vec3(global_coor_light) - fromIsxToLight.origin);
				fromIsxToLight.origin += fromIsxToLight.direction * 0.0001f;
				for (int i = 0; i < geoms_size; i++)
				{
					Geom & geom = dev_geoms[i];
					if (geom.type == CUBE)
					{
						t = boxIntersectionTest(geom, fromIsxToLight, tmp_intersect, tmp_normal, outside);
					}
					else if (geom.type == SPHERE)
					{
						t = sphereIntersectionTest(geom, fromIsxToLight, tmp_intersect, tmp_normal, outside);
					}
					if (t > 0.0f && t_min > t)
					{
						t_min = t;
						hit_geom_index = i;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
					}
				}
				//No shadow, continue.
				if ((hit_geom_index == Light_idx)&&(glm::length(intersect_point - glm::vec3(global_coor_light)) < 1e-3f)) {
					Material material = materials[intersection.materialId];
					glm::vec3 materialColor = material.color;
					glm::vec3 wi = glm::normalize(Intersection_Coordinate - glm::vec3(intersect_point));
					float lightTerm = glm::dot(intersection.surfaceNormal, -wi);
					if (lightTerm < 0) lightTerm += 1;
					float abdot = AbsDot(intersection.surfaceNormal, wi);
					glm::vec3 Final_color(
						materialColor[0] * LightColor[0],
						materialColor[1] * LightColor[1],
						materialColor[2] * LightColor[2]
					);
					glm::vec3 screen_color = lightTerm * Final_color * abdot / pdf_li;
					pathSegments[idx].color = screen_color;
				}
				//Shadowed.
				else {
					pathSegments[idx].color = glm::vec3(0.f);//pathSegments[idx].color * (float(iter) - 1.0f) / float(iter);
				}
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.f);//pathSegments[idx].color * (float(iter) - 1.0f) / float(iter);
		}
	}
}
//======================================================================================================
#define Point2f glm::vec2
#define Point3f glm::vec3
#define PiOver4 0.78539816339744830961f
__device__ glm::vec3 squareToDiskConcentric(const Point2f &sample)
{
	//TODO
	float r, angle;
	float a = 2 * sample[0] - 1;
	float b = 2 * sample[1] - 1;

	if (a > ((-1)*b))
	{
		if (a > b)
		{
			r = a;
			angle = PiOver4 * (b / a);
		}
		else
		{
			r = b;
			angle = PiOver4 * (2.0f - (a / b));
		}
	}
	else
	{
		if (a < b)
		{
			r = (-1) * a;
			angle = PiOver4 * (4 + b / a);
		}
		else
		{
			r = (-1) * b;
			if (!(fabs(b - 0.0f) < 1e-5))
				angle = PiOver4 * (6.0f - (a / b));
			else
				angle = 0;
		}
	}

	glm::vec3 result(r * cos(angle), r * sin(angle), 0);
	return result;
}
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
#if DIRECTLIGHT
		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
#else
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#endif
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	#if REALISTIC_CAMERA
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		float ndc_x = (2 * (x) / float(cam.resolution.x) - 1.0f) * (-1.0f);
		float ndc_y = 1.f - 2.f * y / float(cam.resolution.y);
		float near_clip = 0.01f;
		float far_clip = 100.0f;
		float fovy = cam.fov[1];
		float focalDistance = 15.5f;
		float lensRadius = 1.0f;
		glm::vec3 ndc_point = glm::vec3(ndc_x, ndc_y, 0.f);

		glm::mat4 presp_mat = glm::perspective(glm::radians(fovy), cam.resolution.x / (float)cam.resolution.y, near_clip, far_clip);
		glm::mat4 inv_persp_mat = glm::inverse(presp_mat);
		glm::vec4 camera_point_vec4 = inv_persp_mat * glm::vec4(ndc_point, 1.f);
		glm::vec3 camera_point = glm::vec3(camera_point_vec4);
		Ray r;
		r.origin = glm::vec3(0.0f);
		r.direction = glm::normalize(camera_point);

		float IOR = 1.5f;
		float templensRadius = focalDistance * (IOR - 1.f);

		if (lensRadius > 0)
		{
			//Sample point on lens
			Point2f pLens = Point2f(templensRadius * squareToDiskConcentric(glm::vec2(u01(rng),u01(rng))));
			//Compute point on plane of focus
			float ft = focalDistance / fabs(r.direction[2]);
			Point3f pFocus = r.origin + r.direction * ft;
			//Update ray for effect of lens 375
			r.origin = Point3f(pLens.x, pLens.y, 0);
			r.direction = glm::normalize(pFocus - r.origin);
		}
		glm::mat4 cameraToWorld = glm::lookAt(cam.position, cam.lookAt, cam.up);
		r.origin = glm::vec3(glm::inverse(cameraToWorld) * glm::vec4(r.origin, 1.f));
		r.direction = glm::normalize(glm::vec3(glm::vec3(glm::inverse(cameraToWorld) * glm::vec4(r.direction, 0.f))));
		segment.ray = r;
	#endif
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
	, int iter
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

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
__device__ glm::mat4 Rebuild(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
}
__global__ void move(Geom *geom) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index == 6) {
		geom[6].translation += glm::vec3(0.005f,0.03f,0.01f);
		geom[6].transform = Rebuild(geom[6].translation, geom[6].rotation, geom[6].scale);
		geom[6].inverseTransform = glm::inverse(geom[6].transform);
		geom[6].invTranspose = glm::transpose(geom[6].inverseTransform);
	}
	__syncthreads();
}
__global__ void material_id_generate(int *material_id, ShadeableIntersection* dev_intersection,int numpath) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numpath) {
		material_id[index] = dev_intersection[index].materialId;
	}
	__syncthreads();
}

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
#if MOVINGTRACE
	dim3 mm = (num_paths + blockSize1d - 1) / blockSize1d;
	move << <mm, blockSize1d >> >(dev_geoms);
#endif
#if DIRECTLIGHT
	int Geom_Sum = hst_scene->geoms.size();
	int *Lightidx_all = new int[Geom_Sum];
	int total_light = 0;
	for (int i = 0; i < Geom_Sum; ++i)
	{
		int material_id = hst_scene->geoms[i].materialid;
		if (hst_scene->materials[material_id].emittance > 0) {
			Lightidx_all[total_light] = i;
			total_light++;
		}
	}
	cudaMemcpy(dev_LightIndex, Lightidx_all, total_light * sizeof(int), cudaMemcpyHostToDevice);
	delete(Lightidx_all);
#endif
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if CACHEFIRST
		if ((iter != 1) && (depth == 0)) {
			cudaMemcpy(dev_intersections, dev_cache_intersections, pcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
#endif
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, iter
				);
#if CACHEFIRST
		}
		if ((iter == 1) && (depth == 0)) {
			cudaMemcpy(dev_cache_intersections, dev_intersections, pcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
#endif

#if SORTMATERIAL
		thrust::device_ptr<int> dev_thrust_material_id(material_id);
		thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
		thrust::device_ptr<ShadeableIntersection> dev_inter_inter(dev_intersections);
		material_id_generate << <numblocksPathSegmentTracing, blockSize1d >> > (material_id, dev_intersections, num_paths);
		thrust::sort_by_key(dev_thrust_material_id, dev_thrust_material_id + num_paths, dev_thrust_paths);
		material_id_generate << <numblocksPathSegmentTracing, blockSize1d >> > (material_id, dev_intersections, num_paths);
		thrust::sort_by_key(dev_thrust_material_id, dev_thrust_material_id + num_paths, dev_inter_inter);
#endif
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

		// shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
		// 		iter, num_paths, dev_intersections, dev_paths, dev_materials);

		// working
#if DIRECTLIGHT
		DirectLight_Count++;
		shadeDirectLight << <numblocksPathSegmentTracing, blockSize1d >> >(
			iter,
			num_paths,
			dev_intersections, dev_paths, dev_materials, dev_LightIndex, dev_geoms, total_light, Geom_Sum);
		iterationComplete = true;
#else
#if TIMERCOUNT
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
#endif
		shadeAllMaterial << <numblocksPathSegmentTracing, blockSize1d >> >(
			iter, num_paths, dev_intersections, dev_paths, dev_materials);
#if TIMERCOUNT
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		if ((iter == 1)&&(depth <= 10))
			cout << milliseconds << endl;
#endif
		gatherAndTerminate << <numblocksPathSegmentTracing, blockSize1d >> >(
			num_paths, dev_image, dev_paths, dev_intersections);

		scatterAliveRays << <numblocksPathSegmentTracing, blockSize1d >> >(
			iter, depth, num_paths, dev_paths, dev_intersections,
			dev_materials);

		PathSegment *dev_path_end_new = thrust::remove_if(thrust::device,
			dev_paths, dev_paths + num_paths, AlivePathTest());

		if (depth >= traceDepth || dev_path_end_new == dev_paths) {
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}
		else {
			num_paths = dev_path_end_new - dev_paths;
		}
#endif
		//if (iter == 1)
		//	cout << num_paths << endl;
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