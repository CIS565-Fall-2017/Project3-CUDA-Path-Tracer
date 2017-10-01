#pragma once

#include <glm/gtx/rotate_vector.hpp>
#include "intersections.h"

enum materials_t { diffuse, reflection, refraction, emission, glossy, anisotropic};

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
glm::vec3 calculateRandomDirectionRefraction(
		glm::vec3 normal, glm::vec3 ray, float ior, bool outside, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float p;

	p = u01(rng);

	float _ior = outside ? 1.0f / ior : ior;
	float r0 = powf((1.0f - _ior) / (1.0f + _ior), 2);
	float c = powf(1.0f - abs(glm::dot(ray, normal)), 5);
	float rs = r0 + (1 - r0) * c;
	if (rs > p) { 
		return glm::normalize(glm::reflect(ray, normal));
	}
	else {
		return glm::normalize(glm::refract(ray, normal, _ior));
	}
}


__host__ __device__
glm::vec3 calculateRandomDirectionSpecular(
	glm::vec3 normal, glm::vec3 ray, float exp, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float ts = acos(powf(u01(rng), (1.0 / (exp + 1.0))));
	float ps = 2.0 * PI * u01(rng);
	float c_ps = cos(ps);
	float c_ts = cos(ts);
	float s_ps = sin(ps);
	float s_ts = sin(ts);

	glm::vec3 sdir(c_ps * s_ts, s_ps * s_ts, c_ts);
	glm::vec3 ref = glm::vec3(glm::normalize(glm::reflect(ray, normal)));
	glm::vec3 ax = glm::vec3(glm::normalize(glm::cross(glm::vec3(0, 0, 1), ref)));

	return glm::rotate(sdir, acos(glm::dot(ref, glm::vec3(0, 0, 1))), ax);
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
	bool outside,
	const Material &m,
	thrust::default_random_engine &rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	materials_t mid;
	glm::vec3 col;
	glm::vec3 dir;
	glm::vec3 org;

	float ior = m.indexOfRefraction;
	float spec = m.specular.exponent;
	float emitProb = m.emittance;
	float refractProb = m.hasRefractive;
	float reflectProb = m.hasReflective;
	float diffuseProb = 1 - reflectProb - refractProb;

	if (emitProb > 0.f) mid = emission;
	else if (diffuseProb == 1.f && spec == 0.f) mid = diffuse;
	else if (refractProb == 1.f && spec == 0.f) mid = refraction;
	else if (reflectProb == 1.f && spec == 0.f) mid = reflection;
	else if (reflectProb == 1.f && spec >  0.f) mid = glossy;
	else
	{
		thrust::uniform_real_distribution<float> u01(0, 1);
		float rfl = reflectProb;
		float rfr = rfl + refractProb;
		float rnd = u01(rng);

		if (rnd >= 0 && rnd < rfl) {
			if (spec == 0.f) mid = reflection;
			else mid = glossy;
		}
		else if (rnd >= rfl && rnd < rfr) {
			mid = refraction;
		}
		else {
			mid = diffuse;
		}
	}

	float prob, df_ratio;
	glm::vec3 r;

	switch (mid)
	{
	case anisotropic:
		break;
	case diffuse:
		dir = calculateRandomDirectionInHemisphere(normal, rng);
		col = m.color;
		pathSegment.remainingBounces--;
		break;
	case reflection:
		dir = pathSegment.ray.direction - 2.0f * normal * (glm::dot(pathSegment.ray.direction, normal));
		col = m.specular.color * m.color;
		pathSegment.remainingBounces--;
		break;
	case refraction:
		r = pathSegment.ray.direction;
		dir = calculateRandomDirectionRefraction(normal, r, ior, outside, rng);
		col = m.specular.color * m.color;
		pathSegment.remainingBounces--;
		break;
	case emission:
		col = m.color * m.emittance;
		pathSegment.remainingBounces = 0;
		break;
	case glossy:
		r = pathSegment.ray.direction;
		dir = calculateRandomDirectionSpecular(normal, r, spec, rng);
		col = m.specular.color;
		pathSegment.remainingBounces--; 
		break;
	default:
		dir = calculateRandomDirectionInHemisphere(normal, rng);
		col = m.color;
		pathSegment.remainingBounces--;
		break;
	}

	pathSegment.ray.direction = dir;
	pathSegment.ray.origin = intersect + 0.01f * glm::normalize(pathSegment.ray.direction);
	pathSegment.color *= col;
}
