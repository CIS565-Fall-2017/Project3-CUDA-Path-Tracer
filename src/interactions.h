#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 * I believe this is in worldspace
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine &rng) 
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

__host__ __device__ glm::vec3 f_Lambert(const Material &m, glm::vec3& wo, glm::vec3& wi)
{
	return m.color*INVPI;
}

__host__ __device__ float pdf_Lambert(glm::vec3& wo, glm::vec3& wi)
{
	return utilityCore::AbsCosTheta(wi)*INVPI;
}

__host__ __device__ glm::vec3 sample_f_Lambert(const Material &m, thrust::default_random_engine &rng, 
												glm::vec3& wo, glm::vec3& wi, float& pdf)
{
	wi = calculateRandomDirectionInHemisphere(m.normal, rng);

	if (wo.z < 0.0f)
	{
		wi.z *= -1.0f;
	}
	pdf = pdf_Lambert(wo, wi);

	return f_Lambert(m, wo, wi);
}

__host__ __device__ glm::vec3 f_SpecularBRDF(glm::vec3& wo, glm::vec3& wi)
{
	return Color3f(0.f); //This is because we will assume that ωi has a
						 //zero percent chance of being randomly set to the exact mirror of ωo by
						 //any other BxDF's Sample_f, hence a PDF of zero.
}

__host__ __device__ float pdf_SpecularBRDF(glm::vec3& wo, glm::vec3& wi)
{
	return 0.0f; //This is because we will assume that ωi has a
				 //zero percent chance of being randomly set to the exact mirror of ωo by
				 //any other BxDF's Sample_f, hence a PDF of zero.
}

__host__ __device__ glm::vec3 sample_f_SpecularBRDF(const Material &m, glm::vec3& wo, glm::vec3& wi, float& pdf)
{
	wi = Vector3f(-wo.x, -wo.y, wo.z);
	pdf = 1.0f;
	Color3f color = m.color;//DO FRENEL STUFF FOR ACCURATE COLORS
	return color;
}



__host__ __device__ bool MatchesFlags(BxDFType sampledbxdf)
{
	if ( (sampledbxdf & BSDF_SPECULAR_BRDF) ||
		 (sampledbxdf & BSDF_SPECULAR_BTDF) ||
		 (sampledbxdf & BSDF_LAMBERT)        )
	{
		return true;
	}
}

__host__ __device__ int BxDFsMatchingFlags(const Material &m)
{
	BxDFType flags = BSDF_ALL;
	int num = 0;

	for (int i = 0; i < m.numBxDFs; ++i)
	{
		if (MatchesFlags(m.bxdfs[i]))
		{
			++num;
		}
	}

	return num;
}

__host__ __device__ glm::vec3 sample_f(const Material &m, thrust::default_random_engine &rng,
									glm::vec3& woW, glm::vec3& wiW, float& pdf, BxDFType& sampledType)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float xi0 = u01(rng);
	float xi1 = u01(rng);

	//----------------------------------------------

	//choose which BxDF to sample
	int matchingComps = BxDFsMatchingFlags(m);
	if (matchingComps == 0)
	{
		pdf = 0.0f; //because we don't have a material to sample
		return Color3f(0.0f); //return black
	}

	//----------------------------------------------

	// random bxdf choosing
	BxDFType bxdf;
	int comp = glm::min((int)glm::floor(xi0 * matchingComps), matchingComps - 1);	
	int count = comp;

	for (int i = 0; i<m.numBxDFs; ++i)
	{
		// count is only decremented when a bxdfs[i] == mathcing flag
		if ( MatchesFlags(m.bxdfs[i]) && count-- == 0)
		{			
			bxdf = m.bxdfs[i];
			break;
		}
	}

	//----------------------------------------------

	//Remap BxDF sample u to [0,1)
	Point2f uRemapped = Point2f(glm::min(xi0 * matchingComps - comp, OneMinusEpsilon), xi1);
	//so that we can sample the entire hemisphere and not just a subsection of it;
	//xi[1] is independent of the bxdf, and so the bxdf cant bias it

	//----------------------------------------------

	//Sample chosen BxDF
	Vector3f wo = m.worldToTangent * woW;
	Vector3f wi = Vector3f(0.0f);
	pdf = 0.0f;
	if (sampledType)
	{
		//this is because its a recurring call; we are shooting a ray that bounces off
		//many materials on its, journey, so if it was sampled before, then sampledType
		//will exist and we have to reequate it to the type of the material hit
		sampledType = BSDF_ALL;
	}

	Color3f Color = Color3f(0.0f);
	if (bxdf == BSDF_SPECULAR_BRDF)
	{
		Color = sample_f_SpecularBRDF(m, wo, wi, pdf);
	}
	else if (bxdf == BSDF_LAMBERT)
	{
		Color = sample_f_Lambert(m, rng, wo, wi, pdf);
	}
		
	if (pdf == 0)
	{
		//we dont interact with the material sampled, ie pdf is zero, therefore color returned should be black
		return Color3f(0.0f);
	}
	wiW = m.tangentToWorld * wi;

	//-------------------------------------------------------------------------
	//compute overall pdf with all matching BxDFs

	if ( (bxdf != BSDF_SPECULAR_BRDF) && (matchingComps>1) )
	{
		for (int i = 0; i<m.numBxDFs; ++i)
		{
			if ( m.bxdfs[i] != bxdf && MatchesFlags(m.bxdfs[i]) )
			{
				//overall pdf!!!! so get all pdfs for the
				//different bxdfs and average it out for the bsdf
				if (bxdf == BSDF_SPECULAR_BRDF)
				{
					pdf += pdf_SpecularBRDF(wo, wi);
				}
				else if (bxdf == BSDF_LAMBERT)
				{
					pdf += pdf_Lambert(wo, wi);
				}
			}
		}
	}
	if (matchingComps > 1)
	{
		pdf /= matchingComps;
	}

	//-------------------------------------------------------------------------
	//compute value of BSDF for sampled direction
	if( (bxdf != BSDF_SPECULAR_BRDF) && (matchingComps>1) )
	{
		bool reflect = (glm::dot(wiW, m.normal) * glm::dot(woW, m.normal)) > 0;
		Color = Color3f(0.0); //because the material is reflective or
							  //transmissive so doesn't have its own color
		for (int i = 0; i<m.numBxDFs; ++i)
		{
			if ( MatchesFlags(m.bxdfs[i]) &&
				 ( (reflect && m.reflective) ||
				   (!reflect && m.refractive) ) )
			{
				if (bxdf == BSDF_SPECULAR_BRDF)
				{
					Color += f_SpecularBRDF(wo, wi);
				}
				else if (bxdf == BSDF_LAMBERT)
				{
					Color += f_Lambert(m, wo, wi);
				}				
			}
		}
	}

	return Color;
}

__host__ __device__ Ray spawnNewRay(const ShadeableIntersection& intersection, Vector3f wiW)
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

__host__ __device__ void scatterRay(PathSegment & pathSegment, const ShadeableIntersection& intersection,
									const Material &m, thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	glm::vec3 woW = -pathSegment.ray.direction;
	glm::vec3 wiW;
	float pdf;
	BxDFType sampledType;
	glm::vec3 f = Color3f(0.0f);

	f = sample_f(m, rng, woW, wiW, pdf, sampledType); //returns a color, sets wi, sets pdf, sets sampledType
	if (pdf == 0.0f)
	{
		pathSegment.color = Color3f(0.0f);
	}

	//set up new ray direction
	pathSegment.ray = spawnNewRay(intersection, wiW);

	float absDot = utilityCore::AbsDot(wiW, intersection.surfaceNormal);
	Color3f emittedLight = m.emittance*m.color;

	pathSegment.color = (emittedLight + f*pathSegment.color*absDot)/pdf;
	pathSegment.remainingBounces--;
}

/*
__host__ __device__ void scatterRayOLD( PathSegment & pathSegment,
									glm::vec3 intersect,
									glm::vec3 normal,
									const Material &m,
									thrust::default_random_engine &rng) 
{
	// Update the ray and color associated with the pathSegment
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	glm::vec3 wo = pathSegment.ray.direction;
	glm::vec3 wi = glm::vec3(0.0f);
	glm::vec3 color = pathSegment.color;
	float pdf = 1.0f;
	
	// If the material indicates that the object was a light, "light" the ray
	if (m.emittance > 0.0f)
	{
		pathSegment.color *= m.color *m.emittance*pdf;
		//pathSegment.color = (m.color * m.emittance); //debug view of light bouncing and hitting the light
		pathSegment.remainingBounces = 0; //equivalent of breaking out of the thread
		return;
	}

	if (m.hasReflective == 1)
	{
		//specular component
		wi = glm::reflect(wo, normal);
		color = color*m.specular.color;
		pdf = 0.0f;
	}
	else
	{
		//diffuse
		wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));

		pdf = INVPI;
		//Update Color
		color = color*m.color;// *glm::vec3(INVPI)*(1.0f / pdf);
	}
	
	glm::vec3 originOffset = normal*2.0f*EPSILON;
	originOffset = glm::dot(wi, normal) > 0 ? originOffset : -originOffset;

	//if (pdf == 0.0f)
	//{
	//	color = m.color;
	//	return;
	//}

	//set up new ray direction
	pathSegment.ray.origin = intersect + originOffset;
	pathSegment.ray.direction = wi;

	color *= glm::abs(glm::dot(wi, normal));
	pathSegment.color = color;

	pathSegment.remainingBounces--;
}
*/