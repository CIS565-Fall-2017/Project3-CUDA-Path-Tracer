#pragma once

#include "intersections.h"
#include <glm/gtx/rotate_vector.hpp>

#define FRS 1


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
glm::vec3 calculateRandomDirectionInHemisphereStratified(
	glm::vec3 normal, glm::vec2 sample) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(sample.y); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = sample.x * TWO_PI;

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

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec2 squareToDiskConcentric(glm::vec2 sample) {
	float r, theta;
	float sx = 2 * sample.x - 1;
	float sy = 2 * sample.y - 1;

	if (sx == 0.f && sy == 0.f) {
		return glm::vec2(0.f);
	}

	if (sx >= -sy) {
		if (sx > sy) {
			r = sx;
			if (sy > 0.f) {
				theta = sy / r;
			}
			else {
				theta = 8.f + sy / r;
			}
		}
		else {
			r = sy;
			theta = 2.f - sx / r;
		}
	}
	else {
		if (sx <= sy) {
			r = -sx;
			theta = 4.f - sy / r;
		}
		else {
			r = -sy;
			theta = 6.f + sx / r;
		}
	}

	theta *= PI / 4.f;
	return glm::vec2(r * cos(theta), r * sin(theta));
}

__host__ __device__
glm::vec3 squareToHemisphereCosine(
	glm::vec3 normal, glm::vec2 sample) {
	glm::vec2 sample_2 = squareToDiskConcentric(sample);
	float c = sqrt(1 - sample_2.x*sample_2.x - sample_2.y*sample_2.y);

	glm::vec3 sample3 = glm::vec3(sample_2, c);


	const glm::vec3 up = glm::vec3(0, 0, 1); 
	glm::mat4 rotmat = glm::orientation(normal, up);
	return glm::vec3(rotmat * glm::vec4(sample3, 0));
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
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	float prob = u01(rng);
	glm::vec3 direction;

	if (prob < m.hasReflective)
	{//Reflective
		direction = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.ray.origin = intersect + direction * 1e-3f;
		pathSegment.color *= m.specular.color;
	}
	else if (prob < m.hasReflective + m.hasRefractive)
	{//refractive
		float NI = glm::dot(normal, pathSegment.ray.direction);
		float ratio = m.indexOfRefraction;
		if (NI < 0)
		{
			ratio = 1.f / ratio;
		}
#if FRS
		float r0 = (1.f - ratio) / (1.f + ratio);
		r0 *= r0;
		float x = 1.f + NI;
		float r = r0 + (1.f - r0) * x * x * x * x * x;
		if (u01(rng) < r) 
		{
			direction = glm::reflect(pathSegment.ray.direction, normal);
		}
		else
		{
			direction = glm::refract(pathSegment.ray.direction, normal, ratio);
		}
#else
		direction = glm::refract(pathSegment.ray.direction, normal, ratio);
		/*float temp = 1 - ratio * ratio * (1 - NI * NI);
		if (temp < 0)  // reflection
		{
			direction = glm::reflect(pathSegment.ray.direction, normal);
		}
		else
		{
			direction = glm::refract(pathSegment.ray.direction, normal, ratio);
		}*/
#endif
		pathSegment.color *= m.specular.color;
		pathSegment.ray.origin = intersect + direction * 1e-3f;
	}
	else
	{//defuse
		direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.origin = intersect + direction * EPSILON;
	}
	pathSegment.ray.direction = glm::normalize(direction);
	pathSegment.color *= m.color;
	//pathSegment.color *= glm::abs(glm::dot(pathSegment.ray.direction, normal)) * m.color;
}

__host__ __device__
void scatterRayStratified(
	PathSegment & pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material &m,
	glm::vec2 sample) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	float prob = sample.x;
	glm::vec3 direction;

	if (prob < m.hasReflective)
	{//Reflective
		direction = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.ray.origin = intersect + direction * 1e-3f;
		pathSegment.color *= m.specular.color;
	}
	else if (prob < m.hasReflective + m.hasRefractive)
	{//refractive
		float NI = glm::dot(normal, pathSegment.ray.direction);
		float ratio = m.indexOfRefraction;
		if (NI < 0)
		{
			ratio = 1.f / ratio;
		}
#if FRS
		float r0 = (1.f - ratio) / (1.f + ratio);
		r0 *= r0;
		float x = 1.f + NI;
		float r = r0 + (1.f - r0) * x * x * x * x * x;
		if (sample.y < r)
		{
			direction = glm::reflect(pathSegment.ray.direction, normal);
		}
		else
		{
			direction = glm::refract(pathSegment.ray.direction, normal, ratio);
		}
#else
		direction = glm::refract(pathSegment.ray.direction, normal, ratio);
		/*float temp = 1 - ratio * ratio * (1 - NI * NI);
		if (temp < 0)  // reflection
		{
		direction = glm::reflect(pathSegment.ray.direction, normal);
		}
		else
		{
		direction = glm::refract(pathSegment.ray.direction, normal, ratio);
		}*/
#endif
		pathSegment.color *= m.specular.color;
		pathSegment.ray.origin = intersect + direction * 1e-3f;
	}
	else
	{
		direction = squareToHemisphereCosine(normal, sample);
		pathSegment.ray.origin = intersect + direction * EPSILON;
	}
	pathSegment.ray.direction = glm::normalize(direction);
	pathSegment.color *= m.color;

}

__host__ __device__
glm::vec3 sampleonlight(Geom & light, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	if (light.type == SPHERE)
	{
		float z = 1 - 2 * u01(rng);
		float x = cos(2 * PI*u01(rng))*sqrt(1 - z*z);
		float y = sin(2 * PI*u01(rng))*sqrt(1 - z*z);
		return glm::vec3(light.transform * glm::vec4(x, y, z, 1.f));
	}
	else if (light.type == CUBE)
	{
		float x = u01(rng) - 0.5f;
		float y = u01(rng) - 0.5f;
		float z = u01(rng) - 0.5f;
		return glm::vec3(light.transform * glm::vec4(x, y, z, 1.f));
	}
	return glm::vec3(0.f);
}