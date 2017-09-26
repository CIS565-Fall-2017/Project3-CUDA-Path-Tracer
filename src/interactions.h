#pragma once

#include "intersections.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

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
	
	//hit the light, terminate the path
	int pixelIndex = pathSegment.pixelIndex;
	if (m.emittance > 0) {
		pathSegment.color *= m.color * m.emittance;
		pathSegment.remainingBounces = 0;
	}
	else {		
		pathSegment.remainingBounces--;
		if (pathSegment.remainingBounces == 0) {
			pathSegment.color = glm::vec3(0);
		}
		else {
			if (false && m.hasReflective) {
				//Create a random number between 0 to 1
				thrust::uniform_real_distribution<float> u01(0, 1);
				float probability = u01(rng);
				//if random number is smaller than 0.5, then use reflective brdf.
				//if random number is bigger than 0.5, then use diffusive brdf
				if (probability < 0.5) {
					//reflective
					glm::vec3 ri = pathSegment.ray.direction;
					pathSegment.ray.direction = pathSegment.ray.direction - 2.0f *normal*(glm::dot(pathSegment.ray.direction, normal));
					pathSegment.color /= 0.5;
				}
				else {
					pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
					float lightTerm = glm::abs(glm::dot(normal, pathSegment.ray.direction));
					pathSegment.color *= (m.color) / 0.5f;
				}
				pathSegment.ray.origin = intersect;
			}
			else {
				//Test on pure-diffusive first	
				//Below code does not really work well
				pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
				//float f = 1.0 / PI; //Does not affect result
				//float pdf = 1.0 / (PI); //Does not affect result
				float lightTerm = glm::abs(glm::dot(normal, pathSegment.ray.direction));
				pathSegment.color *= m.color;
				pathSegment.ray.origin = intersect+ pathSegment.ray.direction*0.001f;
			}
		}	
	}
}
