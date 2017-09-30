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
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

	// get probabilty
	thrust::uniform_real_distribution<float> u01(0, 1);
	float probability = u01(rng);
	glm::vec3 incidentDirection = pathSegment.ray.direction;
	glm::vec3 newDirection;

	// ----------------- Pure Specular reflection -------------------------
	if (m.hasReflective == 1.f && m.hasRefractive == 0.f) {
		newDirection = Reflect(-incidentDirection, normal);
		newDirection = glm::normalize(newDirection);
		pathSegment.ray.direction = newDirection;
		pathSegment.ray.origin = intersect;
		pathSegment.color *= m.specular.color; 
	}

	// ----------------- Pure Specular refraction ----------------------------
	else if (m.hasReflective == 0.f && m.hasRefractive == 1.f) {

		bool entering = glm::dot(incidentDirection, normal) < 0;
		float indexOfRefraction = m.indexOfRefraction;
		float eta = entering ? (1.0f / indexOfRefraction) : indexOfRefraction;

		//glm::vec3 newDirection = glm::refract(incidentDirection, normal, eta);
		//pathSegment.ray.direction = newDirection;
		//pathSegment.ray.origin = intersect + 0.0002f * newDirection;
		//pathSegment.color *= m.specular.color;

		glm::vec3 wt;
		if (!Refract(-incidentDirection, Faceforward(normal, -incidentDirection), eta, &wt)) {
			newDirection = Reflect(-incidentDirection, Faceforward(normal, -incidentDirection));
			newDirection = glm::normalize(newDirection);
			pathSegment.ray.direction = newDirection;
			pathSegment.ray.origin = intersect;
		}
		else {
			pathSegment.ray.direction = glm::normalize(wt);
			pathSegment.ray.origin = intersect + 0.0002f * incidentDirection + 0.0002f * Faceforward(normal, incidentDirection);
		}
		pathSegment.color *= m.specular.color;  // divide the probability to counter the chance
	}

	// ------------------- Glass : Specular reflection & refraction-------------------
	else if (m.hasReflective != 0.f && m.hasRefractive != 0.f) {
		float specularSum = m.hasReflective + m.hasRefractive;
		float reflecProb = m.hasReflective / specularSum;
		float refractProb = 1.0f - reflecProb;

		// Use Schlick's Approximation. No specific Frensel model
		float temp = (1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction);
		float R0 = temp * temp;
		float cosineTheta = AbsDot(incidentDirection, normal);
		float frenselCoefficient = R0 + (1.0f - R0) * (1.0f - cosineTheta) * (1.0f - cosineTheta);

		if (probability < reflecProb) {
			newDirection = Reflect(-incidentDirection, normal);
			newDirection = glm::normalize(newDirection);
			pathSegment.ray.direction = newDirection;
			pathSegment.ray.origin = intersect;
			pathSegment.color *= (frenselCoefficient * m.specular.color / reflecProb); // divide the probability to counter the chance
		}

		// if it's specualr and not reflective
		// then it's Refractive
		else
		{
			bool entering = glm::dot(incidentDirection, normal) < 0;
			float indexOfRefraction = m.indexOfRefraction;
			float eta = entering ? (1.0f / indexOfRefraction) : indexOfRefraction;

			glm::vec3 wt;
			if (!Refract(-incidentDirection, Faceforward(normal, -incidentDirection), eta, &wt)) {
				newDirection = Reflect(-incidentDirection, Faceforward(normal, -incidentDirection));
				newDirection = glm::normalize(newDirection);
				pathSegment.ray.direction = newDirection;
				pathSegment.ray.origin = intersect;
			}
			else {
				pathSegment.ray.direction = glm::normalize(wt);
				pathSegment.ray.origin = intersect + 0.0002f * incidentDirection + 0.0002f * Faceforward(normal, incidentDirection);
			}
			pathSegment.color *= ((1.f - frenselCoefficient) * m.specular.color / refractProb);  // divide the probability to counter the chance
		}
	}

	// ------------------- Non-specular / Diffuse Part ---------------------
	else 
	{
		newDirection = calculateRandomDirectionInHemisphere(normal, rng);
		newDirection = glm::normalize(newDirection);
		pathSegment.ray.direction = newDirection;
		pathSegment.ray.origin = intersect;
		// normal and newDirection should have been normalized 
		// pathSegment.color *= (m.color * AbsDot(normal, newDirection));
		
		// Debug Normal
		//pathSegment.color *= normal;

		pathSegment.color *= m.color;
	}
}
