#pragma once

#include "sampling.h"
#include "lambert.h"
#include "specular.h"

__host__ __device__ void sampleMaterials(const Material &m, Vector3f& wo, Vector3f& normal,
										 Color3f& sampledColor, Vector3f& wi, float& pdf,
										 thrust::default_random_engine &rng)
{
	//simple pick and choose, if you implement a more complex material which requires you to 
	//sample multiple bxDFs then include the samplef thing from the other branch
	for (int i = 0; i < m.numBxDFs; i++)
	{
		if (m.bxdfs[i] == BSDF_LAMBERT)
		{
			sampledColor = sample_Lambert(m, rng, wo, wi, normal, pdf);
		}
		else if (m.bxdfs[i] == BSDF_SPECULAR_BRDF)
		{
			sampledColor = sample_Specular(m, wo, wi, normal, pdf);
		}
	}
}