#pragma once

#include "intersections.h"
#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

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

	// Implement the specular and the diffuse shading
	// Update the color of the sample
	// Generte new direction of the ray
	
	float probability = 1.0f; // Will be used later when slecting between refplection and refraction
	glm::vec3 newRayDirection;
	glm::vec3 finalColor = glm::vec3(1.0f,1.0f,1.0f);

	// SPECULAR REFLECTIVE
	if (m.hasReflective) {
		// Update direction 
		newRayDirection = glm::reflect(pathSegment.ray.direction , normal);

		// Update color
		finalColor = m.specular.color * m.color;
	}
	// DIFFUSE
	else {
		// Update direction
		newRayDirection = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));

		// Update color
		finalColor *= fabs(glm::dot(normal, newRayDirection)) * m.color;
	}

	pathSegment.ray.direction = newRayDirection;
	pathSegment.ray.origin = intersect + EPSILON * normal;
	pathSegment.color *= finalColor;
}

/**
* Square to Disk Uniform mapping function
* Based on the implementation in PBRT and CIS460
*/
__host__ __device__
glm::vec2 squareToDiskUniform(const glm::vec2 point2D) {
	float r = std::sqrt(point2D[0]); //x-axis maps to the radius
	float theta = 2.0f * M_PI * point2D[1]; //y-axis maps to the angle on the disc
	return glm::vec2(r * std::cos(theta), r * std::sin(theta));
}

/**
* Square to Disk Concentric mapping function
* Based on the implementation in PBRT and CIS460
*/
__host__ __device__
glm::vec2 squareToDiskConcentric(const glm::vec2 point2D)
{
	float phi, r, u, v;
	float a = 2.0f * point2D[0] - 1.0f;
	float b = 2.0f * point2D[1] - 1.0f;


	if (a > -b)
	{
		if (a > b) //region 1
		{
			r = a;
			phi = (M_PI / 4.0f) * (b / a);
		}
		else // region 2
		{
			r = b;
			phi = (M_PI / 4.0f) * (2.0f - (a / b));
		}
	}
	else
	{
		if (a < b) // region 3
		{
			r = -a;
			phi = (M_PI / 4.0f) * (4.0f + (b / a));
		}
		else // region 4
		{
			r = -b;
			if (b != 0)
			{
				phi = (M_PI / 4.0f) * (6 - (a / b));
			}
			else
			{
				phi = 0;
			}
		}
	}

	u = r * std::cos(phi);
	v = r * std::sin(phi);
	return glm::vec2(u, v);
}

/**
* Depth of Field
* This implementation assumes a thin lens approximation
* Based on the implementation in PBRT and CIS460
*/
__host__ __device__
void depthOfField(const Camera camera, thrust::default_random_engine& rng, PathSegment& segment) {
	if (camera.lRadius > 0) {
		// Generate a sample
		thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 point2D(u01(rng), u01(rng));
		point2D = squareToDiskConcentric(point2D);

		// Point on the lens
		glm::vec2 pLens = camera.lRadius * point2D;

		// Focal point
		glm::vec3 pFocus = segment.ray.origin + camera.fLength * segment.ray.direction;

		// Update the original ray from camera to start from the lens in the direction of the focal point
		glm::vec3 newOrigin = segment.ray.origin + (camera.up * pLens.y) + (camera.right * pLens.x);
		glm::vec3 newDirection = glm::normalize(pFocus - newOrigin);

		segment.ray.origin = newOrigin;
		segment.ray.direction = newDirection;
	}
	else {
		return;
	}
}

// TODO
// Do Direct lighting calculation for each point. Send in a  variable with all the lights in the scene stored in it to scatterRay function.