#pragma once

#include "sampleShapes.h"
#define COMPEPSILON 0.000000001f

__host__ __device__ bool fequals_Vec(Vector3f& v1, Vector3f& v2)
{
	if ((glm::abs(v1.x - v2.x) < COMPEPSILON) &&
		(glm::abs(v1.y - v2.y) < COMPEPSILON) &&
		(glm::abs(v1.z - v2.z) < COMPEPSILON))
	{
		return true;
	}
	return false;
}

__host__ __device__ float sampleLightPDF(Material &m, Vector3f& normal, Vector3f& wi,
										Vector3f& refPoint, Geom& geom)
{
	float light_pdf = 0.0f;
	return light_pdf;
}

__host__ __device__ Color3f sampleLights(Material &m, Vector3f& normal, Vector3f& wi,
										Vector2f &xi, float& pdf, Vector3f& refPoint, Geom& geom)
{
	//ONLY SUPPORTING SPHERE LIGHTS
	GeomType lightShape = geom.type;
	ShadeableIntersection intersection = sampleShapes(xi, pdf, refPoint, geom, lightShape);

	if (pdf == 0.0f || fequals_Vec(refPoint, intersection.intersectPoint))
	{
		return Color3f(0.0f);
	}
	wi = glm::normalize(intersection.intersectPoint - refPoint);

	Color3f emittedLight = m.color*m.emittance; //assumes light is two sided
	return emittedLight;
}