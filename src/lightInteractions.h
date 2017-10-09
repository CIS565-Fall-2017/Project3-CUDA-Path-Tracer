#pragma once

#include "sampleShapes.h"
#include "common.h"

__host__ __device__ Color3f L(const Material &m, const Vector3f& intersectionNormal, const Vector3f &w)
{
	//assumes light is two sided
	return m.emittance*m.color;

	//if (glm::dot(intersectionNormal, w)>0.0f)
	//{
	//	return m.emittance*m.color;
	//}
	//else
	//{
	//	return Color3f(0.f);
	//}
}

__host__ __device__ float sampleLightPDF(Material &m, Vector3f& wi,
										ShadeableIntersection& ref, 
										Geom& geom)
{
	float light_pdf = 0.0f;
	GeomType lightShape = geom.type;
	//we are calculating the solid angle subtended by the light source
	// 1/((cosTheta/r*r)*area) of the solid angle subtended by this is the pdf of the light source
	ShadeableIntersection isect;
	Ray ray = spawnNewRay(ref, wi);
	computeIntersectionOfRayWithSelectedObject(ray, geom, isect);

	if (isect.t < 0.0f)
	{
		return 0.0f;
	}

	float dist = glm::distance(ref.intersectPoint, isect.intersectPoint);

	float _cosTheta = glm::abs(glm::dot(isect.surfaceNormal, -wi));
	light_pdf = dist*dist / (_cosTheta*sampleShapeArea(geom, lightShape));

	return light_pdf;
}

__host__ __device__ Color3f sampleLights(Material &m, Vector3f& normal, Vector3f& wi,
										Vector2f &xi, float& pdf, Vector3f& refPoint, Geom& geom)
{
	//ONLY SUPPORTING SPHERE and SQUAREPLANE LIGHTS
	GeomType lightShape = geom.type;
	ShadeableIntersection intersection = sampleShapes(xi, pdf, refPoint, normal, geom, lightShape);

	if (pdf == 0.0f || fequals_Vec(refPoint, intersection.intersectPoint))
	{
		return Color3f(0.0f);
	}
	wi = glm::normalize(intersection.intersectPoint - refPoint);

	Color3f emittedLight = L(m, intersection.surfaceNormal, -wi);
	return emittedLight;
}