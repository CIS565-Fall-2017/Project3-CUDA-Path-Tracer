#pragma once

#include "intersections.h"

// CHECKITOUT
/**
* Computes a cosine-weighted random direction in a hemisphere.
* Used for diffuse lighting.
*/
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal, thrust::default_random_engine &rng) {
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

#define INVPI 0.31830988618379067154f

__host__ __device__
bool SameHemisphere(glm::vec3 &w, glm::vec3 &wp)
{
	return w.z * wp.z > 0;
}

__host__ __device__ float CosTheta(glm::vec3 wi, glm::vec3 n) {
	return glm::dot(n, wi);
}

__host__ __device__ float AbsCosTheta(glm::vec3 wi, glm::vec3 n) {
	return glm::abs(CosTheta(wi, n));
}
__host__ __device__
float getPdf(glm::vec3 &wo, glm::vec3 &wi, glm::vec3 &n)
{
	return SameHemisphere(wo, wi) ? AbsCosTheta(wi, n) * INVPI : 0;
}

__host__ __device__
float AbsDot(const glm::vec3 n, const glm::vec3 wi)
{
	return glm::abs(glm::dot(n, wi));
}

__host__ __device__
glm::vec3 fresnelDielectric(glm::vec3 &wo, glm::vec3 &wi, glm::vec3 &normal, float etaI, float etaT)
{
	float cosThetaI = glm::clamp(CosTheta(wi, normal), -1.f, 1.f);

	bool entering = cosThetaI > 0.f;
	if (!entering) {
		float temp = etaI;
		etaI = etaT;
		etaT = temp;

		cosThetaI = glm::abs(cosThetaI);
	}

	// Snell's law
	float sinThetaI = glm::sqrt(glm::max(0.f, 1 - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;

	// Total internal reflection
	if (sinThetaT >= 1.f) {
		return glm::vec3(1.f);
	}

	float cosThetaT = glm::sqrt(glm::max(0.f, 1 - sinThetaT * sinThetaT));

	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

	float fr = Rparl * Rparl;
	fr += Rperp * Rperp;
	fr /= 2.f;

	return glm::vec3(fr);
}

__host__ __device__
bool Refract(glm::vec3 &wi, glm::vec3 &n, float eta, glm::vec3 &wt)
{
	// Compute cos theta using Snell's law
	float cosThetaI = glm::dot(n, wi);
	float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = std::sqrt(1 - sin2ThetaT);
	wt = eta * -wi + (eta * cosThetaI - cosThetaT) * glm::vec3(n);

	return true;
}

__host__ __device__
glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v)
{
	return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__
void spawnRay(PathSegment &pathSegment, const glm::vec3 &normal, const glm::vec3 &wi, const glm::vec3 &intersect)
{
	glm::vec3 originOffset = normal * EPSILON;
	originOffset = (glm::dot(wi, normal) > 0) ? originOffset : -originOffset;
	pathSegment.ray.origin = intersect + originOffset;
	pathSegment.ray.direction = wi;
}

/**
* Scatter a ray with some probabilities according to the material properties.
* For example, a diffuse surface scatters in a cosine-weighted hemisphere.
* A perfect specular surface scatters in the reflected ray direction.
* In order to apply multiple effects to one surface, probabilistically choose
* between them.
*
* The visual effect you want is to straight-up add the diffuse and specular
* components. You can do this in a few ways. This logic also applies to
* combining other types of materias (such as refractive).
*
* - Always take an even (50/50) split between a each effect (a diffuse bounce
*   and a specular bounce), but divide the resulting color of either branch
*   by its probability (0.5), to counteract the chance (0.5) of the branch
*   being taken.
*   - This way is inefficient, but serves as a good starting point - it
*     converges slowly, especially for pure-diffuse or pure-specular.
* - Pick the split based on the intensity of each material color, and divide
*   branch result by that branch's probability (whatever probability you use).
*
* This method applies its changes to the Ray parameter `ray` in place.
* It also modifies the color `color` of the ray in place.
*
* You may need to change the parameter list for your purposes!
*/
__host__ __device__
void scatterRay(
	PathSegment & pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng)
{
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	glm::vec3 wo = -pathSegment.ray.direction;
	glm::vec3 wi(0.f);
	glm::vec3 color(1.f);
	float pdf;

	// Reflective Surface
	if (m.hasReflective) {
		wi = glm::reflect(-wo, normal);

		pdf = 1.f;
		color *= m.specular.color;

		// Set up ray direction for next bounce
		spawnRay(pathSegment, normal, wi, intersect);

		// Update color
		pathSegment.color *= m.color * color;
	}
	// Refractive Surface
	else if (m.hasRefractive) {
		// Needa fix PBRT implementation
		/*bool entering = CosTheta(-wo, normal) > 0;

		float etaA = 1.f;
		float etaB = m.indexOfRefraction;
		float etaI = entering ? etaA : etaB;
		float etaT = entering ? etaB : etaA;

		if (!Refract(-wo, Faceforward(glm::vec3(0, 0, 1), -wo), etaI / etaT, wi)) {
			pdf = 0.f;
			color = glm::vec3(0.f);
		}
		else {
			pdf = 1.f;
			color = m.specular.color * (glm::vec3(1.f) - fresnelDielectric(-wo, wi, normal, etaI, etaT)) / AbsCosTheta(wi, normal);
		}*/

		float n1 = 1.f;					// air
		float n2 = m.indexOfRefraction;	// material

		// CosTheta > 0 --> ray outside
		// CosTheta < 0 --> ray inside
		bool entering = CosTheta(normal, -wo) > 0;
		if (!entering) {
			n2 = 1.f / m.indexOfRefraction;
		}

		// Schlick's Approximation
		float r0 = powf((1 - n1) / (1 + n2), 2.f);
		float rTheta = r0 + (1 - r0) * powf(1 - CosTheta(-wo, normal), 5.f);
		
		// Snell's Law
		wi = glm::normalize(glm::refract(-wo, normal, n2));
		// spawnRay() doesn't work?
		pathSegment.ray.direction = wi;
		// Update color
		//pathSegment.color *= m.speculr.color * (glm::vec3(1.f) - fresnelDielectric(-wo, wi, normal, n1, n2)) / AbsCosTheta(wi, normal);
		pathSegment.color *= m.specular.color;
	}
	// Diffuse Surface
	else {
		wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		
		pdf = getPdf(wo, wi, normal);
		//color *= INVPI;

		//if (pdf > 0.f) {
		//	pathSegment.color /= pdf;
		//}
		//else {
		//	pathSegment.color = glm::vec3(0.f);
		//}

		// Set up ray direction for next bounce
		spawnRay(pathSegment, normal, wi, intersect);

		// Update color
		pathSegment.color *= m.color * color;
	}

	pathSegment.color *= AbsDot(normal, wi);
}


// http://corysimon.github.io/articles/uniformdistn-on-sphere/
__host__ __device__
glm::vec3 SphereSample(thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float theta = 2.f * PI * u01(rng);
	float phi = acos(1.f - 2.f * u01(rng));
	float x = sin(phi) * cos(theta);
	float y = sin(phi) * sin(theta);
	float z = cos(phi);

	return glm::vec3(x, y, z);
}

// TODO
// Diffuse Area Light
__host__ __device__
glm::vec3 Sample_Li(const ShadeableIntersection &ref, 
					const glm::vec3 &lightPos,
					glm::vec3 &wi,
					float &pdf,
					thrust::default_random_engine &rng)
{
	wi = glm::normalize(lightPos - ref.intersectPoint);

	glm::vec3 samplePoint = SphereSample(rng);
	if (samplePoint == ref.intersectPoint) {
		return glm::vec3(0.f);
	}

	return glm::vec3(0.f);
}