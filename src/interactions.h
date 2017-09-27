#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
#define pushoutTolerance 0.01f;
namespace Global {
	typedef thrust::tuple<PathSegment, ShadeableIntersection> Tuple;
	class MaterialCmp
	{
	public:
		__host__ __device__ bool operator()(const Tuple &a, const Tuple &b)
		{
			return a.get<1>().materialId < b.get<1>().materialId;
		}
	};
	__host__ __device__ float cosWeightedHemispherePdf(
		const glm::vec3 &normal, const glm::vec3 &dir)
	{
		return fmax(0.f, glm::dot(normal, dir)) / PI;
	}
	__host__ __device__
		glm::vec3 calculateRandomDirectionInHemisphere(
			glm::vec3 normal, thrust::default_random_engine &rng) {
		thrust::uniform_real_distribution<float> u01(0, 1);

		float up = sqrt(u01(rng)); // cos(theta)
		float over = sqrt(1 - up * up); // sin(theta)
		float around = u01(rng) * TWO_PI;

		// Find a direction that is not the normal based off of whether or not the
		// normal's components are all equal to sqrt(1/3) or whether or not at
		// least one component is less than sqrt(1/3). Learned this trick from
		// Peter Kutz.

		glm::vec3 directionNotNormal;
		if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
			directionNotNormal = glm::vec3(1, 0, 0);
		}
		else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
			directionNotNormal = glm::vec3(0, 1, 0);
		}
		else {
			directionNotNormal = glm::vec3(0, 0, 1);
		}

		// Use not-normal direction to generate two perpendicular directions
		glm::vec3 perpendicularDirection1 =
			glm::normalize(glm::cross(normal, directionNotNormal));
		glm::vec3 perpendicularDirection2 =
			glm::normalize(glm::cross(normal, perpendicularDirection1));
		glm::vec3 dir = glm::normalize(up * normal
			+ cos(around) * over * perpendicularDirection1
			+ sin(around) * over * perpendicularDirection2);
		return dir;
	}
}


/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
namespace BSDF {
	__host__ __device__ float LambertPDF(const glm::vec3 &normal, const glm::vec3 &dir) {
		return Global::cosWeightedHemispherePdf(normal, dir);
	}
	__host__ __device__ glm::vec3 LambertSample_f(const glm::vec3& normal, glm::vec3& woW, 
		float& pdf, thrust::default_random_engine &rng,const glm::vec3& mc) {
		woW = Global::calculateRandomDirectionInHemisphere(normal, rng);
		pdf = LambertPDF(normal, woW);
		return mc / PI;
	}
	__host__ __device__ float SpecularPDF() {
		return 0.0f;
	}
	__host__ __device__ glm::vec3 SpecularSample_f(const glm::vec3& wiW,const glm::vec3& normal, glm::vec3& woW,
		float& pdf, thrust::default_random_engine &rng, const glm::vec3& mc) {
		woW = glm::normalize(glm::reflect(wiW, normal));
		pdf = SpecularPDF();
		return mc;
	}
	__host__ __device__ float GlassPDF() {
		return 0.0f;
	}
	__host__ __device__ glm::vec3 GlassBRDF(const glm::vec3) { 
		return glm::vec3(); 
	}
}



__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	//lambert
	float pdf = 0;
	glm::vec3 f;
	glm::vec3 wiW = pathSegment.ray.direction;
	glm::vec3 woW;
	if (m.hasReflective) {
		//pathSegment.ray.direction= glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
		f = BSDF::SpecularSample_f(wiW, normal, woW, pdf, rng, m.color);
		pdf = 1;
	}
	//else if (m.hasRefractive) {
	//	float fresnel;
	//	float cosi, cost;
	//	float etai, etat;
	//	glm::vec3 reflectDir, refractDir;
	//	glm::vec3 isecPt;

	//	isecPt = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
	//	cosi = glm::dot(pathSegment.ray.direction, intersection.surfaceNormal);
	//	etai = 1.f;
	//	etat = material.indexOfRefraction;

	//	if (cosi > 0.f)
	//	{
	//		MyUtilities::swap(etai, etat);
	//		intersection.surfaceNormal = -intersection.surfaceNormal;
	//	}

	//	reflectDir = glm::normalize(glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal));
	//	refractDir = glm::normalize(glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, etai / etat));
	//	cost = glm::dot(refractDir, intersection.surfaceNormal);
	//	fresnel = Fresnel::frDiel(fabs(cosi), etai, fabs(cost), etat);

	//	float rn = u01(rng);
	//	if (rn < fresnel || glm::length2(refractDir) < FLT_EPSILON) // deal with total internal reflection
	//	{
	//		pathSegments[idx].ray = { isecPt + 1e-4f * intersection.surfaceNormal, reflectDir };
	//		pathSegments[idx].misWeight *= materialColor;
	//	}
	//	else
	//	{
	//		pathSegments[idx].ray = { isecPt - 1e-4f * intersection.surfaceNormal, refractDir };
	//		pathSegments[idx].misWeight *= materialColor;
	//	}
	//}
	else {
		f = BSDF::LambertSample_f(normal, woW, pdf, rng, m.color);
	}
	pathSegment.ray.direction = woW;
	pathSegment.ray.origin = intersect + pathSegment.ray.direction*pushoutTolerance;
	pathSegment.color *= f * fmax(0.f, glm::dot(normal, pathSegment.ray.direction)) / (pdf + FLT_EPSILON);
	

}
