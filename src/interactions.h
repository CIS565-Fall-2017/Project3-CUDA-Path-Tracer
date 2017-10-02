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

__host__ __device__ float schlick(float costheta, float n1, float n2)
{
	float R0 = (n1 - n2) / (n1 + n2);
	R0 *= R0;
	return R0 + (1 - R0) * pow((1 - costheta), 5);
}

__host__ __device__ void reflect(
	PathSegment & pathSegment,
	glm::vec3 intersect,
	glm::vec3 &normal,
	const Material &m
)
{
	pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
	pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
	pathSegment.color *= m.color;
	pathSegment.remainingBounces--;
}

__host__ __device__ void refract(
	PathSegment & pathSegment,
	glm::vec3 intersect,
	glm::vec3 &normal,
	const Material &m,
	thrust::default_random_engine &rng) 
{
	float n1, n2;
	float cosTheta, eta;
	float fresnel;

	n1 = 1.0f;
	n2 = m.indexOfRefraction;
	cosTheta = glm::dot(pathSegment.ray.direction, normal);

	if (cosTheta > .0f)
	{
		normal = -normal;
		eta = n2 / n1;
	}
	else
	{
		eta = n1 / n2;
	}

	thrust::uniform_real_distribution<float> u01(0, 1);
	fresnel = schlick(fabs(cosTheta), n1, n2);
	if (u01(rng) < fresnel)
	{
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.color *= m.color;
	}
	else
	{
		pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, eta);
	}

	pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
	pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
	pathSegment.remainingBounces--;
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
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 &normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	if (glm::dot(pathSegment.ray.direction, normal) > 0.0f && m.hasRefractive <= 0.001f)
	{
		pathSegment.color = glm::vec3(0.0f);
		pathSegment.remainingBounces = 0;
		return;
	}
	if (m.hasReflective > 0.0f)
	{
		reflect(pathSegment, intersect, normal, m);
	}
	else if (m.hasRefractive > 0.0f)
	{
		refract(pathSegment, intersect, normal, m, rng);
	}
	else if (m.emittance > 0.0f)
	{
		pathSegment.color *= m.color * m.emittance;
		pathSegment.remainingBounces = 0;
	}
	else
	{

		//PathSegment temp = pathSegment;

		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
		pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
		pathSegment.color *= m.color;
		pathSegment.remainingBounces--;
	}

}
