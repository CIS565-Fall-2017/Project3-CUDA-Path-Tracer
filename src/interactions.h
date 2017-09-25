#pragma once

#include "intersections.h"
#include "materialInteractions.h"

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * Thcan do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's pe visual effect you want is to straight-up add the diffuse and specular
 * components. You robability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

__host__ __device__ Ray spawnNewRay(const ShadeableIntersection& intersection, Vector3f wiW)
{
	Vector3f originOffset = intersection.surfaceNormal * EPSILON;
	// Make sure to flip the direction of the offset so it's in the same general direction as the ray direction
	originOffset = (glm::dot(wiW, intersection.surfaceNormal) > 0) ? originOffset : -originOffset;
	Point3f o(intersection.intersectPoint + originOffset);
	Ray r = Ray();
	r.direction = wiW;
	r.origin = o;
	return r;
}

__host__ __device__ void scatterRay(PathSegment & pathSegment, const ShadeableIntersection& intersection,
									const Material &m, matPropertiesPerIntersection &mproperties, 
									thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	glm::vec3 woW = -pathSegment.ray.direction;
	glm::vec3 wiW;
	float pdf;
	BxDFType sampledType;
	glm::vec3 f = Color3f(0.0f);

	f = sample_f(m, mproperties, rng, woW, wiW, pdf, sampledType); //returns a color, sets wi, sets pdf, sets sampledType

	//set up new ray direction
	pathSegment.ray = spawnNewRay(intersection, wiW);

	if (pdf == 0.0f)
	{
		return;
	}

	float absDot = glm::abs(glm::dot(wiW, intersection.surfaceNormal));
	Color3f emittedLight = m.emittance*m.color;

	pathSegment.color = (emittedLight + f*pathSegment.color*absDot) / pdf;// f / pdf;// 
	pathSegment.remainingBounces--;
}