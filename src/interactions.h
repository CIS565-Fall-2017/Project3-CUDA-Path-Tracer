#pragma once

#include "intersections.h"
#include "sampling.h"
#include "materialInteractions.h"
#include "lightInteractions.h"

__host__ __device__ Ray spawnNewRay(ShadeableIntersection& intersection, Vector3f& wiW)
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

__host__ __device__ void naiveIntegrator(PathSegment & pathSegment,
										 ShadeableIntersection& intersection,
										 Material &m,
										 thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f wo = pathSegment.ray.direction;
	Vector3f wi = glm::vec3(0.0f);
	Vector3f sampledColor = pathSegment.color;
	Vector3f normal = intersection.surfaceNormal;
	float pdf = 0.0f;

	sampleMaterials(m, wo, normal, sampledColor, wi, pdf, rng);

	if (pdf != 0.0f)
	{
		float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
		pathSegment.color *= sampledColor*absdot / pdf;
	}
	
	pathSegment.ray = spawnNewRay(intersection, wi);
	pathSegment.remainingBounces--;
}

__host__ __device__ void directLightingIntegrator(PathSegment & pathSegment,
												  ShadeableIntersection& intersection,
												  Material &m, int num_geoms,
												  Geom* geoms, int num_paths,
												  int &numLights, Light * lights,
												  thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f intersectionPoint = intersection.intersectPoint;
	Vector3f normal = intersection.surfaceNormal;
	Vector3f wo = pathSegment.ray.direction;
	Vector2f xi;
	Vector3f wi;
	Vector3f sampledLightColor;
	float pdf = 0.0f;
	
	//only one light in scene
	int randLightindex = 1;
	int lightindex = lights[randLightindex].lightGeomIndex;
	Geom light = geoms[lightindex];

	thrust::uniform_real_distribution<float> u01(0, 1);
	xi[0] = u01(rng);
	xi[1] = u01(rng);

	// Sample a point on the light
	glm::vec3 samplePoint = SphereSample(xi);
	// Get the light's position based on the sample
	glm::vec3 lightPos(light.transform * glm::vec4(samplePoint, 1.f));

	// Sample the object hit for color
	Color3f f;
	Vector3f f_wi;
	float f_pdf = pdf;
	sampleMaterials(m, wo, normal, f, f_wi, f_pdf, rng);

	// Test for shadows
	PathSegment shadowFeeler;
	shadowFeeler.ray.direction = glm::normalize(lightPos - intersection.intersectPoint);
	shadowFeeler.ray.origin = intersection.intersectPoint + (EPSILON * shadowFeeler.ray.direction);

	ShadeableIntersection isx;
	computeIntersectionsWithSelectedObject(shadowFeeler, geoms, num_geoms, isx);

	Color3f Li = Color3f(0.f);
	// Occluded by object
	if (isx.t > 0.f && m.emittance <= 0.f) 
	{
		pathSegment.color = Color3f(0.f);
	}
	else 
	{
		if (glm::dot(lightPos, shadowFeeler.ray.direction) >= 0.f) 
		{
			Li = m.emittance*m.color;
		}

		pathSegment.color *= f * Li * glm::abs(glm::dot(intersection.surfaceNormal, glm::normalize(shadowFeeler.ray.direction)));
		pathSegment.color *= numLights;
	}

	// Direct lighting only does one bounce
	pathSegment.remainingBounces = 0;
}