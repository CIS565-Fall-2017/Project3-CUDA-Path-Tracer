#pragma once

#include "sampling.h"
#include "lambert.h"
#include "specular.h"
#include "subsurface.h"

__host__ __device__ bool MatchesFlags(const BxDFType sampledbxdf)
{
	if ( (sampledbxdf & BSDF_SPECULAR_BRDF) ||
		 (sampledbxdf & BSDF_SPECULAR_BTDF) ||
		 (sampledbxdf & BSDF_LAMBERT)		||
		 (sampledbxdf & BxDF_SUBSURFACE)    ||
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
									const Vector3f& woW, const Vector3f& normal,
									Geom& geom, Vector3f intersectPoint,
									Color3f& sampledColor, Vector3f& wiW, float& pdf)
{
	BxDFType sampledType;
	thrust::uniform_real_distribution<float> u01(0, 1);

	float xi0 = u01(rng);
	float xi1 = u01(rng);

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
	Color3f Color = Color3f(0.0f);
	bool possibleToSubSurface = false;

	if (bxdf == BxDF_SUBSURFACE)
	{
		//if subsurface stuff cant happen calculate color, pdf, and wi using another bxDF
		Vector3f sample = Vector3f(u01(rng), u01(rng), u01(rng));
		possibleToSubSurface = sample_f_Subsurface(woW, sample, m, geom, normal, intersectPoint, sampledColor, wiW, pdf);

		if (!possibleToSubSurface)
		{
			//change bxDF to another bxDF;
			count = comp;
			for (int i = 0; i<m.numBxDFs; ++i)
			{
				// count is only decremented when a bxdfs[i] == mathcing flag
				if (MatchesFlags(m.bxdfs[i]) && count-- == 0 && m.bxdfs[i] != BxDF_SUBSURFACE)
				{
					bxdf = m.bxdfs[i];
					break;
				}
			}
		}
	}

	if (bxdf == BSDF_SPECULAR_BRDF && !possibleToSubSurface)
	{
		sampledColor = f_Specular(m);
		pdf = pdf_Specular();
		wiW = glm::reflect(woW, normal);
	}
	else if (bxdf == BSDF_LAMBERT && !possibleToSubSurface)
	{
		//sample functions return a color, calculate pdf, and set a new wiW
		wiW = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		pdf = pdf_Lambert(woW, wiW, normal);
		sampledColor = f_Lambert(m, woW, wiW);
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
					pdf += pdf_Specular();
				}
				else if (bxdf == BSDF_LAMBERT)
				{
					pdf += pdf_Lambert(woW, wiW, normal);
				}
				else if (bxdf == BxDF_SUBSURFACE)
				{
					pdf += pdf_Subsurface(woW, wiW, m.thetaMin);
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
					Color += f_Specular(m);
				}
				else if (bxdf == BSDF_LAMBERT)
				{
					Color += f_Lambert(m, woW, wiW);
				}
				else if (bxdf == BxDF_SUBSURFACE)
				{
					float samplePoint = u01(rng);
					Color += f_Subsurface(m, samplePoint);
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
	//sample multiple bxDFs then use the sample_f function
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