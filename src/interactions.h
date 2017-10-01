#pragma once

#include "intersections.h"

#define FRESNEL 0

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
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
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
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

__host__ __device__ float FrDielectric(float cosThetaI, float etaI, float etaT)
{
	cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);
	bool entering = cosThetaI > 0.f;
	//Potentially swap indices of refraction
	if (!entering)
	{
//		std::swap(etaI, etaT);
		float e = etaI;
		etaI = etaT;
		etaT = e;
		cosThetaI = glm::abs(cosThetaI);
	}
	//Compute cosThetaT using Snell's law
	float sinThetaI = glm::sqrt(glm::max(0.0f, 1 - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;
	if (sinThetaT >= 1)
		return 1.0f;
	float cosThetaT = glm::sqrt(glm::max(0.0f, 1 - sinThetaT * sinThetaT));
	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
		((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
		((etaI * cosThetaI) + (etaT * cosThetaT));
	return (Rparl * Rparl + Rperp * Rperp) / 2;
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
	if (m.hasReflective <= 0.0f && m.hasRefractive <= 0.0f)
	{
		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= glm::clamp(glm::dot(glm::normalize(pathSegment.ray.direction), normal), 0.85f, 1.0f);
	}
	else if (m.hasReflective > 0.0f && m.hasRefractive <= 0.0f)
	{
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	}
	else if (m.hasReflective <= 0.0f && m.hasRefractive > 0.0f)
	{
		pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, m.indexOfRefraction);
	}
#if FRESNEL
	else
	{
		float cosThetaI = glm::dot(glm::normalize(pathSegment.ray.direction), normal);
		glm::vec3 fr(FrDielectric(cosThetaI, 1.0f, m.indexOfRefraction));
		thrust::uniform_real_distribution<float> u_sample(0, m.hasReflective + m.hasRefractive);
		float flag = u_sample(rng);
		if (flag < m.hasReflective) 
		{
			//Reflection
			pathSegment.color *= fr * m.specular.color / glm::abs(cosThetaI);
			pathSegment.color = glm::clamp(pathSegment.color, 0.0f, 0.8f);
			pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
		}
		else 
		{
			//Refraction
			pathSegment.color *= (glm::vec3(1.0f) - fr) * m.specular.color / glm::abs(cosThetaI);
			pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, m.indexOfRefraction);
		}
	}
#endif
	pathSegment.ray.origin = intersect + 0.012f * pathSegment.ray.direction;
}
