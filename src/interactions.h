#pragma once

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
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
												  Material* materials, Material &m,
												  Geom* geoms, int numGeoms,
												  Light * lights, int &numLights,
												  thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f intersectionPoint = intersection.intersectPoint;
	Vector3f normal = intersection.surfaceNormal;

	Vector3f wo = pathSegment.ray.direction;
	Vector3f wi = glm::vec3(0.0f);
	float pdf = 0.0f;
	Vector2f xi;

	Vector3f sampledLightColor;
	Vector3f f = Color3f(0.0f);

	//Assuming the scene has atleast one light
	thrust::uniform_real_distribution<float> u01(0, 1);

	int randomLightIndex = std::min((int)std::floor(u01(rng)*numLights), numLights - 1);
	Light selectedLight = lights[randomLightIndex];
	Geom lightGeom = geoms[selectedLight.lightGeomIndex];
	Material lightMat = materials[lightGeom.materialid];

	//sample material of object you hit in the scene
	sampleMaterials(m, wo, normal, f, wi, pdf, rng);

	//Sample Light
	xi = Vector2f(u01(rng), u01(rng));
	Vector3f wi_towardsLight;
	sampledLightColor = sampleLights(lightMat, normal, wi_towardsLight, xi, pdf, intersectionPoint, lightGeom);

	pdf /= numLights;
	if (pdf == 0.0f)
	{
		pathSegment.color = Color3f(0.f);
	}
	else
	{
		float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
		pathSegment.color *= f * sampledLightColor * absdot / pdf;
	}

	//visibility test
	ShadeableIntersection isx;
	computeIntersectionsForASingleRay(pathSegment, geoms, numGeoms, isx);

	// if the shadow feeler ray doesnt hit the sample light then color the pathSegment black
	if (isx.t < 0.0f && isx.hitGeomIndex != selectedLight.lightGeomIndex)
	{
		pathSegment.color = Color3f(0.f);
	}

	pathSegment.remainingBounces = 0;
}