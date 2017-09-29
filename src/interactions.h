#pragma once

#include "intersections.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <thrust/random.h>
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
float calculateLambertPDF(glm::vec3 scatterLightDir, glm::vec3 normal){
	return 1.0 / PI * glm::abs(glm::dot(scatterLightDir, normal));
}

__host__ __device__
float calculateLambertBSDF(glm::vec3 incomingLightDir, glm::vec3 scatterLightDir) {
	return 1.0 / PI;
}

__host__ __device__
float absCos(glm::vec3 light, glm::vec3 normal) {
	return glm::abs(glm::normalize(glm::dot(light, normal)));
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
			//***********************Specular Reflective + Diffusive*************************//
			if (m.hasReflective) {
				//Create a random number between 0 to 1
				thrust::uniform_real_distribution<float> u01(0, 1);
				float probability = u01(rng);
				//if random number is smaller than 0.5, then use reflective brdf.
				//if random number is bigger than 0.5, then use diffusive brdf
				if (probability < 0.5) {
					//reflective
					glm::vec3 ri = pathSegment.ray.direction;
					//pathSegment.ray.direction = pathSegment.ray.direction - 2.0f *normal*(glm::dot(pathSegment.ray.direction, normal));
					pathSegment.ray.direction = glm::reflect(ri, normal);
					pathSegment.color *= m.specular.color / 0.5f;
					pathSegment.ray.origin = intersect + pathSegment.ray.direction*0.001f;
				}
				else {
					//diffusive
					glm::vec3 incomingLightDir = pathSegment.ray.direction;
					glm::vec3 scatterLightDir = calculateRandomDirectionInHemisphere(normal, rng);
					float bsdf = calculateLambertBSDF(incomingLightDir, scatterLightDir);
					float pdf = calculateLambertPDF(scatterLightDir, normal);
					float lightTerm = absCos(normal, scatterLightDir);
					//pathSegment.color *= (m.color*lightTerm*bsdf / pdf) / 0.5f;
					pathSegment.color *= (m.color) / 0.5f;
					pathSegment.ray.direction = scatterLightDir;
					pathSegment.ray.origin = intersect + scatterLightDir*0.001f;
				}			
			}

			//***********************Specular Transmissive + Specular Reflective*************************//
			else if (m.hasRefractive) {
				glm::vec3 incomingLightDir = pathSegment.ray.direction;
				glm::vec3 refractNormal = normal;
				float cosThetaI = glm::clamp(glm::dot(glm::normalize(incomingLightDir), glm::normalize(normal)), -1.0f, 1.0f);
				float etaIn = 1.0f;
				float etaOut = m.indexOfRefraction;
				//Light is leaving the transmissive material
				if (cosThetaI > 0) {
					refractNormal = -normal;
					float temp = etaIn;
					etaIn = etaOut;
					etaOut = temp;
				}
				else {
					cosThetaI = fabs(cosThetaI);
				}

				//Using Snell's Law to compute cosThetaT
				float sinThetaI = std::sqrt(max(0.0f, 1 - cosThetaI*cosThetaI));
				float sinThetaT = etaIn / etaOut*sinThetaI;

				float cosThetaT = std::sqrt(max((float)0, 1 - sinThetaT*sinThetaT));
				float parallelR = ((etaOut*cosThetaI) - (etaIn*cosThetaT))
						/ ((etaOut*cosThetaI) + (etaIn*cosThetaT));
				float perpendicularR = ((etaIn*cosThetaI) - (etaOut*cosThetaT))
						/ ((etaIn*cosThetaI) + (etaOut*cosThetaT));
				float reflectCoefficient = (parallelR*parallelR + perpendicularR*perpendicularR) / 2;				

				//Create a random number between 0 to 1
				thrust::uniform_real_distribution<float> u01(0, 1);
				float probability = u01(rng);
				if (probability < 0.5) {
					//Specular BRDF FresnelDielectric (etaI = 1.0f, etaT = indexOfRefraction)
					glm::vec3 ri = pathSegment.ray.direction;
					glm::vec3 ro = glm::reflect(ri,normal);
					pathSegment.ray.direction = ro;
					glm::vec3 reflectColor = m.specular.color*reflectCoefficient;
					pathSegment.color *= reflectColor / 0.5f;
					pathSegment.ray.origin = intersect + pathSegment.ray.direction*0.001f;
				}
				else {
					//Specular BTDF (etaA = 1.0f, etaB = indexOfRefraction, FresnelDielectric (etaI = 1.0f, etaT = indexOfRefraction))
					glm::vec3 scatterLightDir = glm::vec3(0.f);
					glm::vec3 transmissiveColor = glm::vec3(0.0f);

					//Check total Internal reflection
					if (sinThetaT >= 1) {
						scatterLightDir = glm::reflect(incomingLightDir, refractNormal);
						transmissiveColor = m.specular.color;
					}else {
						scatterLightDir = glm::refract(incomingLightDir, refractNormal, etaIn / etaOut);
						transmissiveColor = m.specular.color*(1 - reflectCoefficient);						
					}
					//float lightTerm = absCos(normal, scatterLightDir);
					pathSegment.color *= transmissiveColor/0.5f;
					pathSegment.ray.direction = scatterLightDir;
					pathSegment.ray.origin = intersect + scatterLightDir*0.001f;
				}								
			}
			
			//***********************Pure Diffusive*************************//
			else {				
				glm::vec3 incomingLightDir = pathSegment.ray.direction; 
				glm::vec3 scatterLightDir = calculateRandomDirectionInHemisphere(normal, rng);
				float bsdf = calculateLambertBSDF(incomingLightDir, scatterLightDir);
				float pdf = calculateLambertPDF(scatterLightDir, normal);
				float lightTerm = glm::abs(glm::dot(normal, scatterLightDir));
				pathSegment.color *= m.color*lightTerm*bsdf/pdf;
				pathSegment.ray.origin = intersect+ scatterLightDir*0.001f;
				pathSegment.ray.direction = scatterLightDir;
			}
		}	
	}
}
