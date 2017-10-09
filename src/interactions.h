#pragma once

#include "intersections.h"
#include <thrust/functional.h>
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
// compute the Phong spectral light.
__host__ __device__ float clamp(float d)
{
	return (d > 0) ? d : 0.0f;
}
// wo, wi are both out of the surface
__host__ __device__ glm::vec3 phongColor(const glm::vec3& wi, const glm::vec3& w0, 
                   const ShadeableIntersection& intersect, const Material& m)
{
	glm::vec3 reflectwi{ glm::reflect(-wi, intersect.surfaceNormal) };
	float amplitude{ glm::dot(reflectwi, w0) };
	amplitude = clamp(amplitude); // may not be necessary
	float component{ powf(amplitude, m.specular.exponent) };
	return component * m.specular.color;
}
// spawn a ray from the point this ray would hit at time t; new ray has newdir
__host__ __device__ void spawn_ray(Ray&  ray, const glm::vec3& normal, const glm::vec3& newdir, float t)
{
	ray.origin += t * ray.direction + EPSILON * normal;
	ray.direction = newdir;
}
__host__ __device__ float numMaterials(const Material& m)
{
	int lambert    { (m.summaryState & MaterialType::Lambert) > 0};
	int reflective { (m.summaryState & MaterialType::Reflective) > 0 };
	int refractive { (m.summaryState & MaterialType::Refractive)  > 0};
	int emissive   { (m.summaryState & MaterialType::Emissive) > 0};
	return  (float) (lambert + reflective + refractive + emissive);
}
// Returns the MaterialType and the Number of materials. When there is more than one material,
// a random one is chosen and the number of materials = 1/(prob of choosing that material) is returned.
// The color needs this to correctly weigh the chances of that material being selected.
__host__ __device__ MaterialType getMaterialType(const Material& m, 
	 thrust::default_random_engine& rng)
{
	int lambert    { (m.summaryState & MaterialType::Lambert)  > 0  };
	int reflective { (m.summaryState & MaterialType::Reflective) > 0 };
	int refractive { (m.summaryState & MaterialType::Refractive) > 0 };
	int emissive   { (m.summaryState & MaterialType::Emissive )  > 0 };
	int numMaterials =  lambert + reflective + refractive + emissive;
    thrust::uniform_int_distribution<int> u(1, numMaterials);
	int memnumber { u(rng)};
	if (lambert && (--memnumber == 0) ) {
		return MaterialType::Lambert;
	}
	if (reflective && (--memnumber == 0) ) {
		return MaterialType::Reflective;
	}
	if (refractive && (--memnumber == 0) ) {
		return MaterialType::Refractive;
	}
	if (emissive && (--memnumber == 0) ) {
		return MaterialType::Emissive;
	}
	return MaterialType::NoMaterial;
}
__host__ __device__ float oneMinusX(float x)
{
	return max(0.0f, 1 - x);
}
__host__ __device__ float evaluateCosTransmitted(float cosI, float eta)
{
	float sinI2 = oneMinusX(cosI * cosI);
	float sinT2 = min(eta* eta * sinI2, 1.0f);
	// at 1.0f or greater Total internal reflection 
	// set sinT2 to 1 and cosT to 0
	return std::sqrt(oneMinusX(sinT2));

}

// my prior implementation of the fresnel coefficient
// eta is the ratio  of incident over transmitted as in glm::refract
__host__ __device__ float evaluateGeneral(float cosThetaI, float eta) 
{
	if (abs(cosThetaI) < EPSILON) {
		return 1.0f;
	}
	float cosT = evaluateCosTransmitted(cosThetaI, eta);
	float t3 = eta * cosThetaI;
	float t4 = eta * cosT;
	float rpll{ (cosThetaI - t4) / (cosThetaI + t4) };
	float rperp{ (t3 - cosT) / (t3 + cosT) };
	float fr = (rpll * rpll + rperp * rperp) * 0.5f;
	return fr;
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
// shader handles refraction and reflection but we could put in the Fresnel
// attenuation
//   Some common cases were removed from this: Now Emissive materials are handled
//   earlier because they terminate the ray.  There still are case that would terminate
//   the ray here:  inside a Lmbertian surface or some No Material (which should never occur).
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        const ShadeableIntersection& intersect,
        const Material &m,
        thrust::default_random_engine &rng) {
        // TODO: implement this.
        // A basic implementation of pure-diffuse shading will just call the
        // calculateRandomDirectionInHemisphere defined above.
        float numM{ numMaterials(m) };

        if (intersect.matl == MaterialType::Lambert)
        {
		
                if (!intersect.outside) {
                    pathSegment.color = glm::vec3(0.0f);
                    pathSegment.remainingBounces = 0;
                }
                else {
          --pathSegment.remainingBounces;
          glm::vec3 wi{ calculateRandomDirectionInHemisphere(
			  intersect.surfaceNormal, rng) };
          glm::vec3 color{ phongColor(wi, 
			  -pathSegment.ray.direction, intersect, m) };
          color += m.color;
          // cos weighted the pdf is cos (theta incident)/(PI * numMaterials); 
          // normally would multiply by the cos (theta I)  but these cancel
          // also the brdf for a constant is R/PI the PIs cancel
          pathSegment.color *= color * numM;
          spawn_ray(pathSegment.ray, intersect.surfaceNormal, wi, intersect.t);
         	}
        }	

        else if (intersect.matl == MaterialType::Reflective)
        {
         	spawn_ray(pathSegment.ray, intersect.surfaceNormal, glm::reflect(pathSegment.ray.direction,
         		intersect.surfaceNormal), intersect.t);
         	pathSegment.color *= m.specular.color * numM;
         	--pathSegment.remainingBounces;
        }
        else if (intersect.matl == MaterialType::Refractive)
        {
        	float ior{ (intersect.outside) ? m.indexOfRefraction : 1 / m.indexOfRefraction };
        	//float costhetaI{ abs(glm::dot(pathSegment.ray.direction, intersect.surfaceNormal)) };
        	pathSegment.color *= m.specular.color * numM;
        	// ior  indexOf refraction is the index away from normal/ index on side of normal.
        	// if the ray came from outside (wo is on the same side of the normal) wi would be 
        	// in the material. glm::refract assumes ior is the incident/transmitted indices or 
        	// if outside far index over the near index since the light wi is coming through the material.
        	spawn_ray(pathSegment.ray, intersect.surfaceNormal, glm::refract(pathSegment.ray.direction,
        		intersect.surfaceNormal, ior), intersect.t);
        	--pathSegment.remainingBounces;
        }
        else { // 
        	pathSegment.remainingBounces = 0;
        	pathSegment.color = glm::vec3(0.0f);
        	
        }

}
// Determine the color of rays that intersect with a light directly or emissive rays.
//  MaterialType must be emissive to call this and the number of remaining Bounces must
//   be greater than 0.
__host__ __device__
void ColorEmissive(
	PathSegment & pathSegment,
	const ShadeableIntersection& intersect,
	const Material &m)
{
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	if (!intersect.outside) {
		pathSegment.color = glm::vec3(0.0f);
		return;
	}
	else
	{
     float numM{ numMaterials(m) };
		pathSegment.color *= m.color *  numM * m.emittance;
	}
        pathSegment.remainingBounces = 0; // terminate the ray
}
// used for sorting matl is the single material that this intersection will use.
// Not used now ... This compiles but I wasn't able to get sort to compile with this.
__host__ __device__ bool thrust::less<ShadeableIntersection>::operator()
    (const ShadeableIntersection &v1, const ShadeableIntersection& v2) const
{
	if (v1.matl != v2.matl) {
		return int(v1.matl) < int(v2.matl);
	}
	else {
		return v1.outside < v2.outside;
	}
}
