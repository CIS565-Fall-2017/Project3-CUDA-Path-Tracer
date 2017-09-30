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