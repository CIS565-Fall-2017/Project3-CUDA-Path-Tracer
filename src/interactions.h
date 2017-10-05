#pragma once

#include "intersections.h"
#include "misHelper.h"

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
	PathSegment & path,
	ShadeableIntersection& isect,
	const Material &m,
	thrust::default_random_engine &rng,
	float& pdf) {

	// Light
	if (m.bsdf == -1) {
		path.color *= (m.color * m.emittance);
		path.remainingBounces = 0;
	}
	// Diffuse
	else if (m.bsdf == 0) {
		//Straight from Line 7 of my NaiveIntegrator.cpp
		const glm::vec3 &n = isect.surfaceNormal;
		const glm::vec3 &wo = -path.ray.direction;

		//This is "f". See Line 7 of LambertBRDF.cpp in my CPU Pathtracer
		glm::vec3 accum_color = m.color * InvPi;

		//This is lamberFactor. See Line 23 of NaiveIntegrator.cpp in my CPU Pathtracer
		float lambert_factor = fabs(glm::dot(n, wo));

		//PDF Calculation
		float dotWo = glm::dot(n, wo);
		float cosTheta = fabs(dotWo) * InvPi;
		pdf = cosTheta;
		
		if (pdf == 0) {
			path.remainingBounces = 0;
			return;
		}

		glm::vec3 integral = (accum_color * lambert_factor)
		                          	/ pdf;
		path.color *= integral;

		//Scatter the Ray
		path.ray.origin = isect.point + n*EPSILON;
		path.ray.direction = calculateRandomDirectionInHemisphere(n, rng);
		path.remainingBounces--;
	} else if (m.bsdf == 1) { //Reflective
		const glm::vec3 &n = isect.surfaceNormal;

		//Scatter the Ray
		path.ray.origin = isect.point + n*EPSILON;
		path.ray.direction = glm::reflect(path.ray.direction, n);
		path.remainingBounces--;
		pdf = 0.f;
	} else if (m.bsdf == 2) { //Refractive
 	    const glm::vec3 n = isect.surfaceNormal;
	    const glm::vec3 wo = -path.ray.direction;
	    
	    //Figure out which way we're going in->out or out -> in
	    //This is needed for incrementing the point along the normal
	    //and refraction
	    const bool entering = glm::dot(wo, n) > 0;
	    const float eta = entering ? 1 / m.indexOfRefraction : m.indexOfRefraction;
	    glm::vec3 faceforwardN = entering ? n : n;
	    
	    //Perform the Refraction
		path.ray.direction = glm::refract(-wo, faceforwardN, eta);

		pdf = 0;
	    
	    //Increment based on whether or not we've changed mediums (!TIR)
	    const bool changedMediums = glm::dot(path.ray.direction, n) < 0.f;
	    const glm::vec3 increment = changedMediums ? -n*EPSILON : n*EPSILON;
	    
	    //Change the path according to calculations
	    path.ray.origin += increment;
		path.color *= m.specular.color * 0.9f;
		path.remainingBounces--;
	} else {
		//SHOULDN'T EVER HAPPEN
	}

	path.ray.direction = glm::normalize(path.ray.direction);
}

//Picks a light at random, then gets the color at that a point on that light
__host__ __device__ glm::vec3 sample_li(const Geom& light, const Material& m, const glm::vec3& ref, thrust::default_random_engine &rng, glm::vec3 *wi, float* pdf_li) {
	if (light.type == CUBE) {
		//SAMPLE SHAPE
		glm::vec3 shape_sample = sampleCube(light, ref, rng, pdf_li);
		*wi = glm::normalize(shape_sample - ref);
		
		if (*pdf_li == 0 || shape_sample == ref) {
			return glm::vec3(0.f);
		}
		
		return m.color * m.emittance;
	} else if (light.type == PLANE) {
		//SAMPLE SHAPE
		glm::vec3 shape_sample = samplePlane(light, ref, rng, pdf_li);
		*wi = glm::normalize(shape_sample - ref);

		if (*pdf_li == 0 || shape_sample == ref) {
			return glm::vec3(0.f);
		}

		return m.color * m.emittance;
	}

	return glm::vec3(0.f);
}