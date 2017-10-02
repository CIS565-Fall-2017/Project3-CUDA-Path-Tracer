#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
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

#define ENABLE_ANTIALISING
#define ENABLE_FILTERING

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

// Hardcoded for now (TODO: expose curve)
__forceinline__
__host__ __device__ glm::vec3 FilmicTonemapping(glm::vec3 x)
{
	return ((x*(0.15f*x + 0.10f*0.50f) + 0.20f*0.02f) / (x*(0.15f*x + 0.50f) + 0.20f*0.30f)) - 0.02f / 0.30f;
}

__global__ void tonemapKernel(glm::vec4 * rawImage, glm::vec4 * resultImage, glm::ivec2 resolution, Film film)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) 
	{
		float maxDistance = glm::max(resolution.x, resolution.y);
		glm::vec2 uv = glm::vec2(x - resolution.x * .5f, y - resolution.y * .5f);
		uv /= maxDistance;

		float vignette = 1.0 - glm::smoothstep(film.vignetteStart, film.vignetteEnd, glm::length(uv) / 0.70710678118f);

		int index = x + (y * resolution.x);
		glm::vec4 pixel = rawImage[index];

		// Divide by accumulated filter weight
		pixel /= pixel.w;
		pixel *= film.exposure * vignette;
		
		// Custom vignette operator
		pixel = glm::mix(pixel * pixel * pixel * .5f, pixel, vignette);

		// Filmic tonemapping reference: http://filmicworlds.com/blog/filmic-tonemapping-operators/
		glm::vec3 current = FilmicTonemapping(glm::vec3(pixel));
		glm::vec3 whiteScale = 1.0f / FilmicTonemapping(glm::vec3(13.2f));
		glm::vec3 color = glm::clamp(current*whiteScale, glm::vec3(0.f), glm::vec3(1.f));

		pixel.x = glm::pow(color.x, film.invGamma);
		pixel.y = glm::pow(color.y, film.invGamma);
		pixel.z = glm::pow(color.z, film.invGamma);

		resultImage[index] = pixel;
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec4* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec4 pix = image[index];

		glm::vec4 pixel;
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

static glm::vec3 * dev_textures = NULL;
static glm::vec4 * dev_image = NULL;
static glm::vec4 * dev_tonemapped_image = NULL;
static char * dev_meshes = NULL;
static int * dev_lights = NULL;

static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static SampledPath * dev_prefiltered = NULL;

static ShadeableIntersection * dev_intersections = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec4));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec4));

	// No need to initialize it
	cudaMalloc(&dev_tonemapped_image, pixelcount * sizeof(glm::vec4));
	
  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_prefiltered, pixelcount * sizeof(SampledPath));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
	
	cudaMalloc(&dev_lights, hst_scene->lights.size() * sizeof(int));
	cudaMemcpy(dev_lights, hst_scene->lights.data(), hst_scene->lights.size() * sizeof(int), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_tonemapped_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

// Hardcoded gaussian filter for ray AA
__forceinline__
__host__ __device__ float evaluateGaussian(float alpha, float radius, float x)
{
	return exp(-alpha * x * x) - exp(-alpha * radius * radius);
}

__forceinline__
__host__ __device__ glm::vec3 palette(float t, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d)
{
	return a + b * glm::cos(6.28318f * (c * t + d));
}

__forceinline__
__host__ __device__ glm::vec3 sampleTexture(glm::vec3 * dev_textures, glm::vec2 uv, TextureDescriptor tex)
{
	uv.y = 1.f - uv.y;
	uv = glm::mod(uv * tex.repeat, glm::vec2(1.f));
	if (tex.type == 0)
	{
		int x = glm::min((int)(uv.x * tex.width), tex.width - 1);
		int y = glm::min((int)(uv.y * tex.height), tex.height - 1);
		int index = y * tex.width + x;
		return dev_textures[tex.index + index];
	}
	else
	{
		int steps = 0;
		glm::vec2 z = glm::vec2(0.f);
		glm::vec2 c = (uv * 2.f - glm::vec2(1.f)) * 1.5f;
		c.x -= .5;

		for (steps = 0; steps < 100; steps++)
		{
			float x = z.x * z.x - z.y * z.y + c.x;
			float y = 2.f * z.x * z.y + c.y;

			z = glm::vec2(x, y);

			if (glm::dot(z, z) > 2.f)
				break;
		}
		
		float sn = float(steps) - log2(log2(dot(z, z))) + 4.0f; // http://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
		sn = glm::clamp(sn, 0.1f, 1.f);
		return glm::vec3(sn);
	}
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, SampledPath * pathSamples, glm::vec3 * dev_textures)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) 
	{
		int index = x + (y * cam.resolution.x);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);

#ifdef ENABLE_ANTIALISING
		thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
#else
		glm::vec2 sample = glm::vec2(0.5f);
#endif
		
		SampledPath & pathSample = pathSamples[index];
		pathSample.position = glm::vec2(x,y) + sample;
		pathSample.color = glm::vec3(0.f);

		PathSegment & segment = pathSegments[index];
		segment.pixelIndex = index;
		segment.color = glm::vec3(1.f);
		segment.remainingBounces = traceDepth;

		segment.ray.origin = cam.position;
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * (((float)x + sample.x) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (((float)y + sample.y) - (float)cam.resolution.y * 0.5f));

		if (cam.aperture > 0.f)
		{
			glm::vec2 bokehUV = glm::vec2(u01(rng), u01(rng));
			glm::vec2 radiusSample = (bokehUV - glm::vec2(.5f)) * cam.aperture; // We need to center it

			// Intersect focal plane
			float d = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
			glm::vec3 focalPoint = segment.ray.origin + (segment.ray.direction * d);

			segment.ray.origin += (cam.right * radiusSample.x) + (cam.up * radiusSample.y);
			segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
			segment.color *= sampleTexture(dev_textures, bokehUV, cam.bokehTexture);
		}
	}
}

__global__ void computeIntersections(int iterations, int depth, int num_paths, PathSegment * pathSegments, Geom * geoms,
	int geoms_size, int * lightIndices, int lightCount, ShadeableIntersection * intersections, char * dev_meshes)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec3 tangent;
		glm::vec2 uv;
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
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, uv, tangent);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, uv, tangent);
			}
			else if (geom.type == MESH)
			{
				t = meshIntersectionTest(geom, pathSegment.ray, dev_meshes, tmp_intersect, tmp_normal, outside, uv, tangent);
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
			intersections[path_index].normal = normal;
			intersections[path_index].tangent = tangent;
			intersections[path_index].uv = uv;

/*
			thrust::default_random_engine rng = makeSeededRandomEngine(iterations, path_index, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);
			float r = u01(rng);

			int lightIndex = glm::clamp((int)r * lightCount, 0, lightCount - 1);
			Geom lightGeom = geoms[lightIndex];*/
/*
			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, uv, tangent);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, uv, tangent);
			}
			else if (geom.type == MESH)
			{
				t = meshIntersectionTest(geom, pathSegment.ray, dev_meshes, tmp_intersect, tmp_normal, outside, uv, tangent);
			}*/
		}
	}
}

__forceinline__
__host__ __device__ glm::vec3 BxDF_Diffuse(glm::vec3 & normal, glm::vec2 & uv, glm::vec3 wo, glm::vec4 sample, glm::vec3 & wi, float & pdf, Material material, glm::vec3 * textureArray)
{
	wi = calculateRandomDirectionInHemisphere(normal, glm::vec2(sample.x, sample.y));
	
	// There's a chance to go through the object
	if (sample.z < material.translucence && sample.w > .5f)
		wi = glm::normalize(-wo - wi * .1f); // Some small randomness

	float cosTheta = glm::abs(glm::dot(normal, wi));
	pdf = cosTheta / glm::pi<float>();

	glm::vec3 result = material.color;

	if (material.diffuseTexture.valid == 1)
		result *= sampleTexture(textureArray, uv, material.diffuseTexture);

	return result * cosTheta * glm::one_over_pi<float>();
}

__forceinline__
__host__ __device__ float FresnelSchlick(float eI, float eT, float HdotV)
{
	float R0 = glm::pow((eI - eT) / (eI + eT), 2.f);
	return R0 + (1.0f - R0) * pow(1.f - HdotV, 5.f);
}

__forceinline__
__host__ __device__ glm::vec3 BxDF_Perfect_Specular_Reflection(glm::vec3 &normal, glm::vec2 &uv, glm::vec3& wo, glm::vec4& sample, glm::vec3 & wi, float & pdf, Material &material, glm::vec3 * textureArray)
{
	wi = glm::reflect(-wo, normal);
	pdf = 1.f;

	glm::vec3 result = material.specular.color;

	if (material.specularTexture.valid == 1)
		result *= sampleTexture(textureArray, uv, material.specularTexture);

	return result;
}

//__forceinline__
__host__ __device__ glm::vec3 BxDF_Perfect_Specular_Refraction(glm::vec3 &normal, glm::vec2 &uv, glm::vec3& wo, glm::vec4& sample, glm::vec3 & wi, float & pdf, Material &material, glm::vec3 * textureArray)
{
	float VdotN = glm::dot(wo, normal);
	bool leaving = VdotN < 0.f;
	glm::vec3 n = normal  *(leaving ? -1.f : 1.f);
	float eta = leaving ?  material.indexOfRefraction : (1.f / material.indexOfRefraction);
	wi = glm::refract(-wo, n, eta);
	pdf = 1.f;

	// Total internal reflection
	if (glm::length(wi) < .01f)
		return glm::vec3(0.f);

	glm::vec3 result = material.specular.color;

	if (material.specularTexture.valid == 1)
		result *= sampleTexture(textureArray, uv, material.specularTexture);
	
	return result;
}

__forceinline__
__host__ __device__ glm::vec3 BxDF_Glass(glm::vec3 &normal, glm::vec2 &uv, glm::vec3& wo, glm::vec4& sample, glm::vec3 & wi, float & pdf, Material &material, glm::vec3 * textureArray)
{
	float VdotN = glm::dot(wo, normal);
	bool leaving = VdotN < 0.f;
	float eI = leaving ? material.indexOfRefraction : 1.f;
	float eT = leaving ? 1.f : material.indexOfRefraction;

	float HdotV = glm::abs(glm::dot(normal, wo));
	float fresnel = FresnelSchlick(eI, eT, HdotV);

	if (sample.z > 0.5f)
		return BxDF_Perfect_Specular_Reflection(normal, uv, wo, sample, wi, pdf, material, textureArray) * fresnel;
	else
		return BxDF_Perfect_Specular_Refraction(normal, uv, wo, sample, wi, pdf, material, textureArray) * (1.f - fresnel);
}

// Uber shader for now...
__global__ void shadeKernel(int iter, int num_paths, ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments, Material * materials, glm::vec3 * textureArray)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];

		glm::vec3 resultColor = pathSegments[idx].color;
		int remainingBounces = pathSegments[idx].remainingBounces;

		// By default, return black and stop
		pathSegments[idx].color = glm::vec3(0.f);
		pathSegments[idx].remainingBounces = 0;

		if (intersection.t > 0.0f && glm::length(resultColor) > EPSILON)
		{
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec4 sample = glm::vec4(u01(rng), u01(rng), u01(rng), u01(rng));
			
			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) 
			{
				pathSegments[idx].color = resultColor * (material.color * material.emittance);
				pathSegments[idx].remainingBounces = 0; // It makes no sense to keep bouncing after a light
			}
			else if(remainingBounces > 1)
			{
				glm::vec3 wo = pathSegments[idx].ray.direction * -1.f;
				glm::vec3 n = intersection.normal;
				glm::vec3 wi;
				float pdf;
				glm::vec3 bxdf = glm::vec3(1.f);

				if (material.normalTexture.valid == 1)
				{
					glm::vec3 tangent = intersection.tangent;
					glm::vec3 bitangent = glm::normalize(glm::cross(n, tangent));

					glm::vec3 texData = sampleTexture(textureArray, intersection.uv, material.normalTexture);
					n = glm::normalize(tangent * texData.r + bitangent * texData.g + n * texData.b);
				}

				if (material.hasRefractive > 0.f && material.hasReflective > 0.f)
					bxdf = BxDF_Glass(n, intersection.uv, wo, sample, wi, pdf, material, textureArray);
				else if (material.hasRefractive > 0.f)
					bxdf = BxDF_Perfect_Specular_Refraction(n, intersection.uv, wo, sample, wi, pdf, material, textureArray);
				else if (material.hasReflective > 0.f)
					bxdf = BxDF_Perfect_Specular_Reflection(n, intersection.uv, wo, sample, wi, pdf, material, textureArray);
				else
					bxdf = BxDF_Diffuse(n, intersection.uv, wo, sample, wi, pdf, material, textureArray);

				if (pdf == 0.f)
					pdf = 1.f;

				resultColor *= bxdf / pdf;

				Ray outRay;
				outRay.origin = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t + wi * .001f;
				outRay.direction = wi;

				pathSegments[idx].ray = outRay;
				pathSegments[idx].color = resultColor;
				pathSegments[idx].remainingBounces = remainingBounces - 1;
			}
		}
	}
}

__global__ void filterKernel(glm::ivec2 resolution, int iterations, glm::vec4 * image, SampledPath * prefiltered, Film film)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y)
	{
		int fromX = glm::max((int)(x - film.filterRadius), 0);
		int toX = glm::min((int) glm::ceil(x + film.filterRadius), resolution.x - 1);

		int fromY = glm::max((int)(y - film.filterRadius), 0);
		int toY = glm::min((int)glm::ceil(y + film.filterRadius), resolution.y - 1);

#ifdef ENABLE_FILTERING
		glm::vec4 accumulated = glm::vec4(0.f);
		glm::vec2 pixelCenter = glm::vec2(x + .5f, y + .5f);

		for (int dy = fromY; dy <= toY; ++dy)
		{
			for (int dx = fromX; dx <= toX; ++dx)
			{
				int index  = dy * resolution.x + dx;
				SampledPath sample = prefiltered[index];
				glm::vec2 offset = (sample.position - pixelCenter) / film.filterRadius;
				float weight = evaluateGaussian(film.filterAlpha, film.filterRadius, offset.x);
				weight *= evaluateGaussian(film.filterAlpha, film.filterRadius, offset.y); // TODO: use a precomputed table
				accumulated += glm::vec4(sample.color * weight, weight);
			}
		}

		int finalIndex = y * resolution.x + x;
		image[finalIndex] += accumulated;
#else
		int finalIndex = y * resolution.x + x;
		image[finalIndex] += glm::vec4(prefiltered[finalIndex].color, 1.f);
#endif
	}
}

// Add the current iteration's output to the overall image
__global__ void scatterKernel(int nPaths, glm::ivec2 resolution, int iterations, SampledPath * sampledPaths, PathSegment * iterationPaths, Film film)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		int pixelIndex = iterationPath.pixelIndex;
		sampledPaths[pixelIndex].color = iterationPath.color;
	}
}

struct PathTerminationOperator
{
	__device__
	bool operator()(const PathSegment x)
	{
		// If there are no remaining bounces or the path does not contribute in any meaningful way.
		return x.remainingBounces > 0 && glm::length(x.color) > .001f;
	}
};

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

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, dev_prefiltered, dev_textures);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	int nonTerminatedPathCount = num_paths;

	bool iterationComplete = false;
	while (!iterationComplete) 
	{
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (nonTerminatedPathCount + blockSize1d - 1) / blockSize1d;
		
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (iter, 
			depth, nonTerminatedPathCount, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_lights, hst_scene->lights.size(), dev_intersections, dev_meshes);

		checkCUDAError("Intersection testing");
		cudaDeviceSynchronize();
		depth++;

		shadeKernel <<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter, nonTerminatedPathCount, dev_intersections, dev_paths, dev_materials, dev_textures);

		cudaDeviceSynchronize();

		// Compact the rays depending on path termination
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, PathTerminationOperator());

		// Recalculate the amount of paths alive
		nonTerminatedPathCount = (dev_path_end - dev_paths);

		iterationComplete = nonTerminatedPathCount == 0 || depth == traceDepth;
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	scatterKernel <<<numBlocksPixels, blockSize1d>>>(num_paths, cam.resolution, iter, dev_prefiltered, dev_paths, hst_scene->state.film);

	// Resolve filtering
	filterKernel << <blocksPerGrid2d, blockSize2d >> >(cam.resolution, iter, dev_image, dev_prefiltered, hst_scene->state.film);

	// Tonemap
	tonemapKernel << <blocksPerGrid2d, blockSize2d >> >(dev_image, dev_tonemapped_image, cam.resolution, hst_scene->state.film);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_tonemapped_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_tonemapped_image, pixelcount * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}


__global__ void correctTexturesKernel(glm::vec3 * texture, glm::vec3 gamma, int size)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < size)
		texture[index] = glm::pow(texture[index], gamma);
}


void initializeDeviceTextures(Scene * scene)
{
	int totalMemory = 0;

	for (int i = 0; i < scene->textures.size(); i++)
		totalMemory += scene->textures[i]->GetWidth() * scene->textures[i]->GetHeight();

	std::vector<int> offsetList;

	if (totalMemory > 0)
	{
		cudaMalloc(&dev_textures, totalMemory * sizeof(glm::vec3));

		const int blockSize1d = 128;

		int offset = 0;
		for (int i = 0; i < scene->textures.size(); i++)
		{
			offsetList.push_back(offset);

			Texture * tex = scene->textures[i];
			int size = tex->GetWidth() * tex->GetHeight();
			cudaMemcpy(dev_textures + offset, tex->GetData(), size * sizeof(glm::vec3), cudaMemcpyHostToDevice);

			glm::vec3 gamma = glm::vec3(tex->GetGamma());
			dim3 numBlocksPixels = (size + blockSize1d - 1) / blockSize1d;
			correctTexturesKernel << <numBlocksPixels, blockSize1d >> > (dev_textures + offset, gamma, size);

			offset += size;
		}
	}

	// Now we need to set all texture descriptor indices
	if(scene->state.camera.bokehTexture.index >= 0)
		scene->state.camera.bokehTexture.index = offsetList[scene->state.camera.bokehTexture.index];

	for (Material & m : scene->materials)
	{
		if(m.diffuseTexture.index >= 0)
			m.diffuseTexture.index = offsetList[m.diffuseTexture.index];

		if (m.specularTexture.index >= 0)
			m.specularTexture.index = offsetList[m.specularTexture.index];

		if (m.normalTexture.index >= 0)
			m.normalTexture.index = offsetList[m.normalTexture.index];
	}

	checkCUDAError("initializeDeviceTextures");
}

void initializeMeshes(Scene * scene)
{
	// Build kd trees and compact memory
	for (int i = 0; i < scene->meshes.size(); i++)
		scene->meshes[i]->Build();

	// Allocate
	int totalMemory = 0;
	for (int i = 0; i < scene->meshes.size(); i++)
		totalMemory += scene->meshes[i]->compactDataSize;

	cudaMalloc(&dev_meshes, totalMemory);
	checkCUDAError("Allocate mesh memory");

	int offset = 0;
	std::vector<int> offsetList;

	// Copy
	for (int i = 0; i < scene->meshes.size(); i++)
	{
		offsetList.push_back(offset);
		Mesh * mesh = scene->meshes[i];
		int size = mesh->compactDataSize;
		int * compactData = mesh->compactNodes;
		cudaMemcpy((void*) ((char*)dev_meshes + offset), compactData, size, cudaMemcpyHostToDevice);
		offset += size;
	}

	// Update references
	for (int i = 0; i < scene->geoms.size(); i++)
	{
		if (scene->geoms[i].type == MESH) 
		{
			Mesh * mesh = scene->meshes[scene->geoms[i].meshData.offset];
			AABB bounds = mesh->meshBounds;
			scene->geoms[i].meshData.maxAABB = bounds.max;
			scene->geoms[i].meshData.minAABB = bounds.min;
			scene->geoms[i].meshData.offset = offsetList[scene->geoms[i].meshData.offset];
		}
	}

	std::cout << "Mesh memory: " << totalMemory << std::endl;

	checkCUDAError("initializeMeshes");
}
