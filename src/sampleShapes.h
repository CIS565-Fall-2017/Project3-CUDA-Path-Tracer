#pragma once

#include "sampling.h"

__host__ __device__ ShadeableIntersection sampleSphere(Vector2f &xi, float& pdf, Geom& geom, float area)
{
	glm::vec4 pObj = glm::vec4(sampling_SquareToSphereUniform(xi), 0.0f); //used to calculate normal

	ShadeableIntersection it;
	it.surfaceNormal = glm::vec3(glm::normalize(geom.invTranspose * pObj));
	pObj.w = 1.0f;
	it.intersectPoint = Point3f(geom.transform * pObj);

	area = 4.f * PI * geom.scale.x * geom.scale.x; // We're assuming uniform scale
	pdf = 1.0f / area;

	return it;
}

__host__ __device__ ShadeableIntersection sampleShapes(Vector2f &xi, float& pdf, Vector3f& refPoint, 
													  Geom& geom, GeomType& lightShape)
{
	//ONLY DOES SPHERES right now
	float area;
	ShadeableIntersection isect;

	if (lightShape == SPHERE)
	{
		isect = sampleSphere(xi, pdf, geom, area);
	}

	Vector3f wi = glm::normalize(isect.intersectPoint - refPoint);
	float absDot = glm::abs(glm::dot(isect.surfaceNormal, -wi));

	if (absDot == 0.0f)
	{
		pdf = 0.0f;
	}
	else
	{
		pdf = glm::distance2(isect.intersectPoint, refPoint) / (absDot*area);
	}

	return isect;
}