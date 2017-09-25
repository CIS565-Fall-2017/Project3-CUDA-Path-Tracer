#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utilities.h"

/**
* Computes a cosine-weighted random direction in a hemisphere.
* Used for diffuse lighting.
* I believe this is in worldspace
*/
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine &rng)
{
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
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
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

__host__ __device__ float AbsCosTheta(const Vector3f &w) { return std::abs(w.z); }

__host__ __device__ glm::vec3 f_Lambert(const Material &m, glm::vec3& wo, glm::vec3& wi)
{
	return m.color*INVPI;
}

__host__ __device__ float pdf_Lambert(glm::vec3& wo, glm::vec3& wi)
{
	return AbsCosTheta(wi)*INVPI;
}

__host__ __device__ glm::vec3 sample_f_Lambert(const Material &m, const matPropertiesPerIntersection &mproperties,
											thrust::default_random_engine &rng, glm::vec3& wo, glm::vec3& wi, float& pdf)
{
	wi = calculateRandomDirectionInHemisphere(mproperties.normal, rng);

	if (wo.z < 0.0f)
	{
		wi.z *= -1.0f;
	}
	pdf = pdf_Lambert(wo, wi);

	return f_Lambert(m, wo, wi);
}
