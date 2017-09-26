#pragma once

#include "intersections.h"
#include "sampling.h"
#include "materialInteractions.h"

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * Thcan do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's pe visual effect you want is to straight-up add the diffuse and specular
 * components. You robability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

__host__ __device__ Ray spawnNewRay(const ShadeableIntersection& intersection, Vector3f& wiW)
{
	Vector3f originOffset = intersection.surfaceNormal * EPSILON;
	// Make sure to flip the direction of the offset so it's in the same general direction as the ray direction
	originOffset = (glm::dot(wiW, intersection.surfaceNormal) > 0) ? originOffset : -originOffset;
	Point3f o(intersection.intersectPoint + originOffset);
	Ray r = Ray();
	r.direction = wiW;
	r.origin = o;
	return r;
}

__host__ __device__ void scatterRay( PathSegment & pathSegment,
									 ShadeableIntersection& intersection,
									 const Material &m,
									 thrust::default_random_engine &rng) 
{
	// Update the ray and color associated with the pathSegment
	glm::vec3 wo = pathSegment.ray.direction;
	glm::vec3 wi = glm::vec3(0.0f);
	glm::vec3 color = pathSegment.color;
	float pdf = 1.0f;

	if (m.hasReflective == 1)
	{
		//specular component
		wi = glm::reflect(wo, intersection.surfaceNormal);
		color = m.specular.color;
		pdf = 0.0f;
	}
	else
	{
		//diffuse
		wi = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng));
		//pdf
		pdf = INVPI*glm::abs(glm::cos(glm::dot(wi, intersection.surfaceNormal)));
		//Update Color
		color = m.color*INVPI;
	}
	
	if (pdf != 0.0f)
	{
		float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
		pathSegment.color *= color*absdot / pdf;
	}

	pathSegment.ray = spawnNewRay(intersection, wi);
	pathSegment.remainingBounces--;
}

__host__ __device__ void naiveIntegrator(PathSegment & pathSegment,
										 ShadeableIntersection& intersection,
										 const Material &m,
										 thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f wo = pathSegment.ray.direction;
	Vector3f wi = glm::vec3(0.0f);
	Vector3f sampledColor = pathSegment.color;
	Vector3f normal = intersection.surfaceNormal;
	float pdf = 0.0f;

	sampleMaterials(m, wo, normal, sampledColor, wi, pdf, rng);

	//if (pdf <= EPSILON)
	//{
	//	pathSegment.ray = spawnNewRay(intersection, wi);
	//	pathSegment.remainingBounces--;
	//}
	if (pdf != 0.0f)
	{
		float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
		pathSegment.color *= sampledColor*absdot / pdf;
	}
	
	pathSegment.ray = spawnNewRay(intersection, wi);
	pathSegment.remainingBounces--;
}

/*
Full Lighting

DirectLighting
BSDFBasedDirectLighting
MIS
IndirectBSDFLighting
UpdateRayDirection

//----------------------- Indirect Lighting (Global Illumination) -----
Vector3f wiW_BSDF_Indirect;
float pdf_BSDF_Indirect;
Point2f xi_BSDF_Indirect = sampler->Get2D();

if( depth == recursionLimit  || flag_CameFromSpecular )
{
	directLightingColor += intersection.Le(woW);
}

flag_CameFromSpecular = false;
if(flag_Hit_Specular) //if( (sampledType & BSDF_SPECULAR) == BSDF_SPECULAR )
{
	flag_CameFromSpecular = true;
}

directLightingColor *= accumulatedThroughputColor;  //for only BSDFIndirectLighting, assume directLighting is 1,1,1

if(!flag_NoBSDF)
{
	//f term
	Color3f f_BSDF_Indirect = intersection.bsdf->Sample_f(woW, &wiW_BSDF_Indirect, xi_BSDF_Indirect,
	&pdf_BSDF_Indirect, BSDF_ALL, &sampledType);

	if(pdf_BSDF_Indirect != 0.0f)
	{
		//No Li term per se, this is accounted for via accumulatedThroughputColor

		//absDot Term
		float absDot_BSDF_Indirect = AbsDot(wiW_BSDF_Indirect, intersection.normalGeometric);

		//LTE term
		accumulatedThroughputColor *= f_BSDF_Indirect * absDot_BSDF_Indirect / pdf_BSDF_Indirect;
	}
}

flag_Terminate = RussianRoulette(accumulatedThroughputColor, probability, depth); //can change accumulatedThroughput
accumulatedRayColor += directLightingColor;

Ray n_ray_BSDF_Indirect = intersection.SpawnRay(wiW_BSDF_Indirect);
woW = -n_ray_BSDF_Indirect.direction;
r = n_ray_BSDF_Indirect;
depth--;
}

return accumulatedRayColor;

*/