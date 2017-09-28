#pragma once

#include "sampling.h"
#include "lambert.h"
#include "specular.h"
#include "sampleShapes.h"

__host__ __device__ Color3f sampleLights(const Material &m, Vector3f& normal, Vector3f& wi, 
										 Vector2f &xi, float& pdf, Vector3f& refPoint, Geom& geom)
{
	//ONLY SUPPORTING SPHERE LIGHTS
	GeomType lightShape = geom.type;
	ShadeableIntersection intersection = sampleShapes(xi, pdf, refPoint, geom, lightShape);

	if (pdf == 0.0f || refPoint == intersection.intersectPoint)
	{
		return Color3f(0.0f);
	}
	wi = glm::normalize(intersection.intersectPoint - refPoint);

	Color3f emittedLight = m.color*m.emittance; //assumes light is two sided
	return emittedLight;
}