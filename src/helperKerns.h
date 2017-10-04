#pragma once

#include <glm/glm.hpp>
#include <cuda.h>

//This would normally be wi.z in my CPU pathtracer, but this is not tangent space!
__host__ __device__ float CosTheta(const glm::vec3& n, const glm::vec3& wi) {
	return glm::abs(glm::dot(n, wi));
}

__host__ __device__ bool SameHemisphere(const glm::vec3& normal, const glm::vec3& wi, const glm::vec3& wo) {
	float dotWi = glm::dot(wi, normal);
	float dotWo = glm::dot(wo, normal);

	return (dotWi >= 0 && dotWo >= 0) || (dotWi < 0 && dotWo < 0);
}