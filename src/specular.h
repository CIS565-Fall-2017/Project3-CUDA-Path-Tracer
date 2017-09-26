#pragma once

#include "sampling.h"

__host__ __device__ Color3f f_Specular(const Material &m)
{
	return m.specular.color;
}

__host__ __device__ float pdf_Specular()
{
	return 0.0f;
}

__host__ __device__ Color3f sample_Specular(const Material &m, Vector3f& wo, Vector3f& wi, Vector3f& normal, float& pdf)
{
	//sample functions return a color, calculate pdf, and set a new wiW
	Color3f color = f_Specular(m);
	pdf = pdf_Specular();
	wi = glm::reflect(wo, normal);

	return color;
}