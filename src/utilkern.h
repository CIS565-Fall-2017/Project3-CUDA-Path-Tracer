#pragma once
#include "glm/glm.hpp"

__host__ __device__ bool isBlack(const glm::vec3 c) {
	return (c.r <= 0.f && c.g <= 0.f && c.b <= 0.f);
}

__host__ __device__ float absDot(const glm::vec3 a, const glm::vec3 b) {
	return glm::abs(glm::dot(a, b));
}

__host__ __device__ float cosTheta(const glm::vec3 a, const glm::vec3 b) {
	return glm::dot(a, b);
}

__host__ __device__ float absCosTheta(const glm::vec3 a, const glm::vec3 b) {
	return glm::abs(glm::dot(a, b));
}

__host__ __device__ bool sameHemisphere(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& n) {
	const float woDotN = glm::dot(wo, n);
	const float wiDotN = glm::dot(wi, n);
	if ( (woDotN > 0.f && wiDotN > 0.f) ||
		 (woDotN < 0.f && wiDotN < 0.f) )
	{
		return true;
	} 
	return false;
}


__host__ __device__ glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
    return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__ void CoordinateSystem(const glm::vec3& norm, 
	glm::vec3& tan, glm::vec3& bit) {
	if (std::abs(norm.x) > std::abs(norm.y)) {
		tan = glm::vec3(-norm.z, 0, norm.x) / std::sqrt(norm.x * norm.x + norm.z * norm.z);
	} else {
		tan = glm::vec3(0, norm.z, -norm.y) / std::sqrt(norm.y * norm.y + norm.z * norm.z);
	}
	bit = glm::cross(norm, tan);
}
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        const glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}
__global__ void copyMaterialIDsToArrays(const int num_paths,
	int* dev_materialIDsForIntersectionsSort,
	int* dev_materialIDsForPathsSort,
	const ShadeableIntersection* dev_intersections) 
{
	const int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid >= num_paths) { return; }
	dev_materialIDsForIntersectionsSort[thid]	= dev_intersections[thid].materialId;
	dev_materialIDsForPathsSort[thid]			= dev_intersections[thid].materialId;
}
