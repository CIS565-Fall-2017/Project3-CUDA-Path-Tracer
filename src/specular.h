#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utilities.h"

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