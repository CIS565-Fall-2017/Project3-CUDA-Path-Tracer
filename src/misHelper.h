#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include "src\sceneStructs.h"
#include <cuda_texture_types.h>


/*********************************************
**********************************************
******        Multiple Importance       ******
******             Sampling             ******
**********************************************
**********************************************/

// This file has lots of important helper kernels for MIS

__device__ __host__ float cubeArea(const Geom& light) {
	return 2 * light.scale.x * light.scale.y *
		   2 * light.scale.z * light.scale.y *
		   2 * light.scale.x * light.scale.z;
}
__device__ __host__ float planeArea(const Geom& light) {
	return light.scale.x * light.scale.y;
}

__device__ __host__ glm::vec3 sampleCube(const Geom& light, const glm::vec3& ref, thrust::default_random_engine &rng, float* pdf) {

	//Get a sample point
	thrust::uniform_real_distribution<float> u_verts(0, 1);
	glm::vec3 sample_li = glm::vec3(u_verts(rng) - 0.5f, 0, u_verts(rng) - 0.5f);
	sample_li = glm::vec3(light.transform * glm::vec4(sample_li, 1));

	glm::vec3 normal_li = glm::vec3(0, 0, -1);

	glm::vec3 wi = glm::normalize(sample_li - ref);

	//Get shape area and convert it to Solid angle
	float cosT = fabs(glm::dot(-wi, normal_li));
	float solid_angle = (glm::length2(sample_li - ref) / cosT);

	*pdf = solid_angle / cubeArea(light);

	//Check if dividing by 0.f
	*pdf = isnan(*pdf) ? 0.f : *pdf;

	return sample_li;
}

__device__ __host__ glm::vec3 samplePlane(const Geom& light, const glm::vec3& ref, thrust::default_random_engine &rng, float* pdf) {

	//Get a sample point
	thrust::uniform_real_distribution<float> u_verts(0,1);
	glm::vec3 sample_li_local = glm::vec3(u_verts(rng) - 0.5f, u_verts(rng) - 0.5f, 0);
	glm::vec3 sample_li = multiplyMV(light.transform, glm::vec4(sample_li_local, 1));

	glm::vec3 wi = glm::normalize(sample_li - ref);

	glm::vec3 normal_li = glm::vec3(light.invTranspose * glm::vec4(0, 0, 1, 0));
	*pdf = 1 / planeArea(light);

	//Get shape area and convert it to Solid angle
	float cosT = glm::abs(glm::dot(-wi, normal_li));
	float solid_angle = (glm::length2(sample_li - ref) / cosT);

	*pdf *= solid_angle;

	return sample_li;
}

__host__ __device__ float power_heuristic(int nf, float fpdf, int ng, float gpdf) {
	float f = nf * fpdf, g = ng * gpdf;
	if (fpdf == 0 && gpdf == 0) return 0.f;

	return (f*f) /
		(f*f + g*g);
}


__host__ __device__ float pdf(int bsdf, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& n) {
	if (bsdf == 0) {
		//PDF Calculation
		float dotWo = glm::dot(n, wo);
		float cosTheta = fabs(dotWo) * InvPi;
		return cosTheta;
	} else if (bsdf == 1 || bsdf == 2) {
		return 0.f;
	}
	else {
		return 0.f;
	}
}

__host__ __device__ glm::vec3 f(const Material m, const glm::vec3& wo, const glm::vec3& wi) {
	int bsdf = m.bsdf;
	if (bsdf == 0) {
		return m.color * InvPi;
	}
	else if (bsdf == 1 || bsdf == 2) {
		return glm::vec3(0.f);
	}
	else {
		return glm::vec3(0.3f);
	}
}

__host__ __device__ bool isBlack(const glm::vec3& vec) {
	return vec[0] == 0 && vec[1] == 0 && vec[2] == 0;
}

///__host__ __device__ float planeIntersectionTest(Geom plane, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal)
__host__ __device__ float pdfLi(const Geom& light,const ShadeableIntersection& ref, const glm::vec3 wi) {
	if (light.type == PLANE) {
		//To Be Filled:
		glm::vec3 isectLightPoint;
		glm::vec3 normal;
		//Input
		Ray ray;
		ray.origin = ref.point;
		ray.direction = wi;

		if (planeIntersectionTest(light, ray, isectLightPoint, normal) < 0) {
			return 0.f;
		}

		return glm::length2(ref.point - isectLightPoint) /
			(glm::abs(glm::dot(normal, -wi)) * planeArea(light));
	} else if (light.type == CUBE) {
			//TODO ?
	}
		return 0.f;
}

__host__ __device__ bool isSpecular(const int& bsdf) {
	return bsdf == 1 || bsdf == 2;
}