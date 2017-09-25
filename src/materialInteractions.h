#pragma once

#include "intersections.h"
#include "lambert.h"
#include "specular.h"

__host__ __device__ bool MatchesFlags(BxDFType sampledbxdf)
{
	if ((sampledbxdf & BSDF_SPECULAR_BRDF) ||
		(sampledbxdf & BSDF_SPECULAR_BTDF) ||
		(sampledbxdf & BSDF_LAMBERT))
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

__host__ __device__ glm::vec3 sample_f(const Material &m, const matPropertiesPerIntersection &mproperties, 
			thrust::default_random_engine &rng, glm::vec3& woW, glm::vec3& wiW, float& pdf, BxDFType& sampledType)
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
		if (MatchesFlags(m.bxdfs[i]) && count-- == 0)
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
	Vector3f wo = mproperties.worldToTangent * woW;
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
		Color = sample_f_Lambert(m, mproperties, rng, wo, wi, pdf);
	}

	if (pdf == 0)
	{
		//we dont interact with the material sampled, ie pdf is zero, therefore color returned should be black
		return Color3f(0.0f);
	}
	wiW = mproperties.tangentToWorld * wi;

	//-------------------------------------------------------------------------
	//compute overall pdf with all matching BxDFs

	if ((bxdf != BSDF_SPECULAR_BRDF) && (matchingComps>1))
	{
		for (int i = 0; i<m.numBxDFs; ++i)
		{
			if (m.bxdfs[i] != bxdf && MatchesFlags(m.bxdfs[i]))
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
	if ((bxdf != BSDF_SPECULAR_BRDF) && (matchingComps>1))
	{
		bool reflect = (glm::dot(wiW, mproperties.normal) * glm::dot(woW, mproperties.normal)) > 0;
		Color = Color3f(0.0); //because the material is reflective or
							  //transmissive so doesn't have its own color
		for (int i = 0; i<m.numBxDFs; ++i)
		{
			if (MatchesFlags(m.bxdfs[i]) &&
				((reflect && m.reflective) ||
				(!reflect && m.refractive)))
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