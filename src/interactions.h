#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine &rng) 
{
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



//SINCE THESE DONT WORK IN UTILITY FUNCTIONS, PUTTING IT HERE

__host__ __device__ 
float AbsDot(const glm::vec3 &a, const glm::vec3 &b)
{
	return glm::abs(glm::dot(a, b));
}

__host__ __device__ 
float CosTheta(const glm::vec3 &wi, const glm::vec3 &n)
{
	//Since dot(wi, n) = |wi| * |n| * cosTheta
	return glm::dot(wi, n) / (glm::length(wi) * glm::length(n));
}

__host__ __device__ 
float AbsCosTheta(const glm::vec3 &wi, const glm::vec3 &n)
{
	return glm::abs(CosTheta(wi, n));
}

__host__ __device__ 
bool SameHemisphere(const glm::vec3 &w, const glm::vec3 &wp)
{
	return w.z * wp.z > 0;
}

__host__ __device__ 
glm::vec3 Refract(const glm::vec3 &wi, const glm::vec3 &normal, float eta)
{
	//Compute cos theta using Snell's law
	float cosThetaI = glm::dot(normal, wi);
	float sin2ThetaI = std::max(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	//Handle total internal reflection for transmission
	if (sin2ThetaT >= 1)	return glm::vec3(0.0f);
	float cosThetaT = std::sqrt(1 - sin2ThetaT);
	return eta * -wi + (eta * cosThetaI - cosThetaT) * normal;
}

//FRESNEL 
//__host__ __device__
//glm::vec3 evaluateFresnelDielectric(float cosThetaI)
//{
//	float clampedCTI = glm::clamp(cosThetaI, -1.0f, 1.0f);
//
//	//potentially swap indices of refraction
//	float _etaI = etaI;     //without saving these private variables here, i got a "read only" error
//	float _etaT = etaT;
//	
//	bool entering = clampedCTI > 0.0f;
//	
//	if (!entering)
//	{
//		float temp = _etaI;
//		_etaI = _etaT;
//		_etaT = temp;
//
//		clampedCTI = std::abs(clampedCTI);
//	}
//
//	//compute cosThetaT using snell's law
//	float sinThetaI = std::sqrt(std::max(float(0), float(1 - clampedCTI * clampedCTI)));
//	float sinThetaT = (_etaI / _etaT) * sinThetaI;
//	
//	// Handle total internal reflection for transmission
//	if (sinThetaT >= 1.0f) return glm::vec3(0.0f);
//	float cosThetaT = std::sqrt(std::max(float(0), 1 - sinThetaT * sinThetaT));
//
//
//	float Rpar1 = ((_etaT * clampedCTI) - (_etaI * cosThetaT)) /
//		((_etaT * clampedCTI) + (_etaI * cosThetaT));
//
//	float Rperp = ((_etaI * clampedCTI) - (_etaT * cosThetaT)) /
//		((_etaI * clampedCTI) + (_etaT * cosThetaT));
//
//	float result = ((Rpar1 * Rpar1) + (Rperp * Rperp)) / 2.0f;
//
//	return glm::vec3(result);
//}




/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materials (such as refractive).
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
        thrust::default_random_engine &rng) 
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.


	glm::vec3 wo = pathSegment.ray.direction;
	//glm::vec3 wo = -pathSegment.ray.direction;

	glm::vec3 wi = glm::vec3(0.0f);

	glm::vec3 diffuseColor = glm::vec3(1.0f);
	glm::vec3 specColor = glm::vec3(1.0f);
	
	//float totalPdf = 0.0f;
	//float specPdf = 0.0f;
	//float diffusePdf = 0.0f;

	//If specular material
	if (glm::length(m.specular.color) > 0)
	{
		//If reflective
		if (m.hasReflective > 0)
		{
			//Calculate wi
			wi = glm::reflect(wo, normal);
			//wi = glm::reflect(-wo, normal);

			specColor = m.specular.color;
			//specColor = (evaluateFresnelDielectric(CosTheta(wi, normal)) * m.specular.color) / AbsCosTheta(wi, normal);

			//specPdf = 1.0f;
		}
		//If refractive
		else if (m.hasRefractive > 0)
		{

		}
		//If both reflective and refractive
		else
		{

		}
	}
	
	//If diffuse material
	else
	{
		//Calculate wi
		//Cosine sample the hemisphere and get a random direction 
		wi = calculateRandomDirectionInHemisphere(normal, rng);
		
		diffuseColor = m.color;
		//diffuseColor = m.color / PI;

		//if (wo.z < 0.0f)	wi.z *= -1.0f;

		//Calculate PDF
		//Since we're doing hemisphere sampling, need to give 
		//higher probability to rays that are close to 90deg since cos(90) = 0
		//if (SameHemisphere(wo, wi))		diffusePdf = AbsCosTheta(wi, normal) / PI; 		
		//else							diffusePdf = 0.0f;
	}

	//Spawn a new ray
	glm::vec3 offset = normal * EPSILON;
	offset = (glm::dot(wi, normal) > 0) ? offset : -offset;
	pathSegment.ray.origin = intersect + offset;
	pathSegment.ray.direction = wi;

	//TESTING
	pathSegment.color *= diffuseColor * specColor;


	//Calculate color based on pdf
	//totalPdf = specPdf + diffusePdf;
	//if (totalPdf == 0.f)
	//{
	//	//This should be result of Le term 
	//	//HOW DO I TAKE CARE OF THIS? IN HERE OR SHADER FUNCTION? 
	//	//THE PATH IM LOOKING AT IN HERE DOESNT HAVE EMITTANCE RIGHT? 
	//	//SO DONT I HAVE TO DO IT IN SHADER?

	//	pathSegment.color = glm::vec3(0.0f);
	//}
	//else
	//{
	//	float absDot = AbsDot(wi, normal);
	//	glm::vec3 newColor = diffuseColor * specColor;
	//	pathSegment.color *= (newColor * absDot) / totalPdf;
	//}

}//end ScatterRay function
