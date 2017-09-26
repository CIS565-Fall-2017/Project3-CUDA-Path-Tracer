#pragma once

#include "sampling.h"

__host__ __device__ float AbsCosTheta(const Vector3f &w, const Vector3f & normal) 
{ 
	return glm::abs(glm::cos(glm::dot(w, normal)));
}

__host__ __device__ Color3f f_Lambert(const Material &m, Vector3f& wo, Vector3f& wi)
{
	return m.color*INVPI;
}

__host__ __device__ float pdf_Lambert(Vector3f& wo, Vector3f& wi, Vector3f& normal)
{
	return AbsCosTheta(wi, normal)*INVPI;
}

__host__ __device__ Color3f sample_Lambert(const Material &m, thrust::default_random_engine &rng, 
										   Vector3f& wo, Vector3f& wi, Vector3f& normal, float& pdf)
{
	//sample functions return a color, calculate pdf, and set a new wiW
	wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
	pdf = pdf_Lambert(wo, wi, normal);
	Color3f color = f_Lambert(m, wo, wi);

	return color;
}