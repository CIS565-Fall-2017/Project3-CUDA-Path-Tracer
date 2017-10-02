#pragma once

#include "intersections.h"

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
#define Shift_Bias 0.02f

__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng,
		bool outside) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	if (glm::dot(pathSegment.ray.direction, normal) > 0.0f && m.hasRefractive <= 0.001f)
	{
		pathSegment.color = glm::vec3(0.0f);
		pathSegment.remainingBounces = 0;
		return;
	}
	if (m.emittance > 0.0f) {
		// emittance
		pathSegment.is_terminated = true;
		pathSegment.remainingBounces = 0;
		pathSegment.color *= (m.color * m.emittance);
		return;
	}
	else if (m.hasReflective > 0.0f) {
		// Reflective
		pathSegment.remainingBounces--;
		pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
		pathSegment.ray.origin = intersect + Shift_Bias * pathSegment.ray.direction;
		pathSegment.color *= m.specular.color;
		pathSegment.color *= glm::abs(glm::dot(pathSegment.ray.direction, normal)) * m.color;
	}
	else if (m.hasRefractive > 0.0f) {
		// Refractive
		pathSegment.remainingBounces--;
		float eta = outside ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
		
		thrust::uniform_real_distribution<float> u01(0, 1);
		float random_flag = u01(rng);

		float cos_theta = abs(glm::dot(pathSegment.ray.direction, normal));
		float R0 = pow((1 - eta) / (1 + eta), 2);
		float fresel = R0 + (1 - R0)*pow(1 - cos_theta, 5);
		if (fresel > random_flag) {
			pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
			pathSegment.ray.origin = intersect + Shift_Bias * pathSegment.ray.direction;
			pathSegment.color *= m.specular.color;
			pathSegment.color *= glm::abs(glm::dot(pathSegment.ray.direction, normal)) * m.color;
		}
		else {
			pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, eta));
			pathSegment.ray.origin = intersect + Shift_Bias * pathSegment.ray.direction;
			pathSegment.color *= m.color;
		}
	}
	else{
		// Diffuse
		pathSegment.remainingBounces--;

		pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		pathSegment.ray.origin = intersect + Shift_Bias * pathSegment.ray.direction;
		//pathSegment.ray.origin = intersect + Shift_Bias * normal;
		//pathSegment.color *= glm::abs(glm::clamp(glm::dot(pathSegment.ray.direction, normal), 0.90f, 1.0f)) * m.color;
		pathSegment.color *= m.color;
	}
}
