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

__host__ __device__
glm::vec3 cosineWeightedSample(glm::vec3 normal, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float u1 = u01(rng);
	float u2 = u01(rng);

	float r = sqrt(u1);
	float theta = TWO_PI * u2;

	float x = r * glm::cos(theta);
	float y = r * glm::sin(theta);
	float z = sqrt(max(0.0f, 1.0f - u1));
	glm::vec3 tangentSpaceSample = glm::normalize(glm::vec3(x, y, z));

	glm::vec3 tangentSpaceNormal(0.0f, 0.0f, 1.0f);
	float angle = glm::acos(glm::dot(tangentSpaceNormal, normal));
	glm::vec3 cross = glm::cross(tangentSpaceNormal, normal);
	glm::vec3 axis = cross / glm::length(cross);

	glm::vec3 v0(0.0f, axis[2], -axis[1]);
	glm::vec3 v1(-axis[2], 0.0f, axis[0]);
	glm::vec3 v2(axis[1], -axis[0], 0.0f);
	glm::mat3 axisSkewed(v0, v1, v2);

	glm::mat3 rotation = glm::mat3(1.0f) + glm::sin(angle)*axisSkewed + ((1.0f - glm::cos(angle)) * axisSkewed * axisSkewed);

	return glm::normalize(rotation * tangentSpaceSample);
}

__host__ __device__
glm::vec3 generateSphereSample(Geom& sphere, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float u1 = u01(rng);
	float u2 = u01(rng);

	float z = 1.0f - 2.0f * u1;
	float r = sqrt(max(0.0f, 1.0f - z * z));
	float phi = TWO_PI * u2;

	glm::vec4 localSample(r * glm::cos(phi), r * glm::sin(phi), z, 1.0f);
	glm::vec4 worldSample = sphere.transform * localSample;
	return glm::vec3(worldSample);
}

__host__ __device__
glm::vec3 generateCubeSample(Geom& cube, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	
	float x = u01(rng) - 0.5f;
	float y = u01(rng) - 0.5f;
	float z = u01(rng) - 0.5f;

	glm::vec4 localSample(x, y, z, 1.0f);
	glm::vec4 worldSample = cube.transform * localSample;
	return glm::vec3(worldSample);
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
glm::vec3 scatterRay(
		PathSegment & pathSegment,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	return cosineWeightedSample(normal, rng);
}
