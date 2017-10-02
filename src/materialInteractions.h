#pragma once

#include "sampling.h"
#include "lambert.h"
#include "specular.h"

__host__ __device__ bool MatchesFlags(const BxDFType sampledbxdf)
{
	if ( (sampledbxdf & BSDF_SPECULAR_BRDF) ||
		 (sampledbxdf & BSDF_SPECULAR_BTDF) ||
		 (sampledbxdf & BSDF_LAMBERT)		||
		 (sampledbxdf & BSDF_GLASS) )
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

__host__ __device__ BxDFType chooseBxDF(const Material &m, thrust::default_random_engine &rng, 
									   Color3f& sampledColor, Vector3f& wiW, float& pdf)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::uniform_real_distribution<float> u02(0, 1);

	float xi0 = u01(rng);
	float xi1 = u02(rng);

	//----------------------------------------------

	//choose which BxDF to sample
	int matchingComps = BxDFsMatchingFlags(m);
	if (matchingComps == 0)
	{
		pdf = 0.0f; //because we don't have a material to sample
		sampledColor = Color3f(0.0f); //return black
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

	return bxdf;
}

__host__ __device__ Color3f sample_f(const Material &m, thrust::default_random_engine &rng,
									Color3f& sampledColor, Vector3f& wiW, float& pdf)
{

	//TODO

	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::uniform_real_distribution<float> u02(0, 1);

	float xi0 = u01(rng);
	float xi1 = u02(rng);

	//----------------------------------------------

	//choose which BxDF to sample
	int matchingComps = BxDFsMatchingFlags(m);
	if (matchingComps == 0)
	{
		pdf = 0.0f; //because we don't have a material to sample
		sampledColor = Color3f(0.0f); //return black
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

	//Remap BxDF sample u to [0,1)
	Point2f uRemapped = Point2f(glm::min(xi0 * matchingComps - comp, OneMinusEpsilon), xi1);
	//so that we can sample the entire hemisphere and not just a subsection of it;
	//xi[1] is independent of the bxdf, and so the bxdf cant bias it

	//----------------------------------------------

	//Sample chosen BxDF
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
		sampledColor = f_Specular(m);
		pdf = pdf_Specular();
		wiW = glm::reflect(wo, normal);
	}
	else if (bxdf == BSDF_LAMBERT)
	{
		//sample functions return a color, calculate pdf, and set a new wiW
		wiW = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		pdf = pdf_Lambert(wo, wiW, normal);
		sampledColor = f_Lambert(m, wo, wiW);
	}

	if (pdf == 0)
	{
		//we dont interact with the material sampled, ie pdf is zero, therefore color returned should be black
		return Color3f(0.0f);
	}

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
					pdf += pdf_Specular(wo, wiW);
				}
				else if (bxdf == BSDF_LAMBERT)
				{
					pdf += pdf_Lambert(wo, wiW);
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
		//bool reflect = (glm::dot(wiW, mproperties.normal) * glm::dot(woW, mproperties.normal)) > 0;
		//Color = Color3f(0.0); //because the material is reflective or
		//					  //transmissive so doesn't have its own color
		for (int i = 0; i<m.numBxDFs; ++i)
		{
			if (m.bxdfs[i] != bxdf && MatchesFlags(m.bxdfs[i]))
			{
				if (bxdf == BSDF_SPECULAR_BRDF)
				{
					Color += f_Specular(wo, wiW);
				}
				else if (bxdf == BSDF_LAMBERT)
				{
					Color += f_Lambert(m, wo, wiW);
				}
			}
		}
	}

	return Color;
}



__host__ __device__ void sampleMaterials(const Material &m, const Vector3f& wo, const Vector3f& normal,
										 Color3f& sampledColor, Vector3f& wi, float& pdf,
										 thrust::default_random_engine &rng)
{
	//simple pick and choose, if you implement a more complex material which requires you to 
	//sample multiple bxDFs then include the samplef thing from the other branch
	BxDFType sampledBxDF = chooseBxDF(m, rng, sampledColor, wi, pdf);

	if (sampledBxDF == BSDF_LAMBERT)
	{
		//sample functions return a color, calculate pdf, and set a new wiW
		wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		pdf = pdf_Lambert(wo, wi, normal);
		sampledColor = f_Lambert(m, wo, wi);
	}
	else if (sampledBxDF == BSDF_SPECULAR_BRDF)
	{
		sampledColor = f_Specular(m);
		pdf = pdf_Specular();
		wi = glm::reflect(wo, normal);
	}
}