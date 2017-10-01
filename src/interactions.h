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

__host__ __device__ float clamp(float d)
{
	return (d > 0) ? d : 0.0;
}
// Zeros the color of the segment and sets intersection time to -1.0f
__host__ __device__ void ZeroSegment(PathSegment& segment, ShadeableIntersection& intersection)
{
	segment.color = glm::vec3(0.0f);
	--segment.remainingBounces;
	intersection.t = -1.0f;
}
// spawn a ray from the point this ray would hit at time t; new ray has newdir
__host__ __device__ void spawn_ray(Ray&  ray, glm::vec3& normal, glm::vec3& newdir, float t)
{
	ray.origin += t * ray.direction + EPSILON * normal;
	ray.direction = newdir;
}
__host__ __device__ MaterialType getMaterialType(const Material& m)
{
	if (m.emittance > 0) {
		return MaterialType::Emmissive;
	}
	else if (m.hasReflective > 0) {
		return MaterialType::Reflective;
	}
	else if (m.hasRefractive > 0) {
		return MaterialType::Refractive;
	}
	else {
		return MaterialType::Lambert;
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
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        ShadeableIntersection& intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	glm::vec3& negwout{ pathSegment.ray.direction };
	float dotP{ -glm::dot(negwout, normal) };
	if (dotP > 0 && intersect.t > 0 && m.hasReflective > 0) {
		pathSegment.color *= m.color;
		spawn_ray(pathSegment.ray, normal, glm::reflect(negwout, normal), intersect.t);
	}
	else if (dotP > 0 && intersect.t > 0) {
		// the pdf for Lambert is coswi/Pi and we divide by that to get
		// the MonteCarlo integral estimate. We also need to multiply by cos wi so these
		// terms cancel out.
		pathSegment.color *= PI * m.color;
		spawn_ray(pathSegment.ray, normal,
			calculateRandomDirectionInHemisphere(normal, rng), intersect.t);
	}
	else {
		ZeroSegment(pathSegment, intersect);
		
	}

}
