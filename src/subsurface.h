#pragma once

#include "sampling.h"

__host__ __device__ float AbsCosTheta(const Vector3f &w, const Vector3f & normal) 
{ 
	return glm::abs(glm::cos(glm::dot(w, normal)));
}

__host__ __device__ Color3f f_Lambert(const Material &m, const Vector3f& wo, const Vector3f& wi)
{
	return m.color*INVPI;
}

__host__ __device__ float pdf_Lambert(const Vector3f& wo, const Vector3f& wi, const Vector3f& normal)
{
	return AbsCosTheta(wi, normal)*INVPI;
}