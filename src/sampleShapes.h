#pragma once

#include "sampling.h"
#include "common.h"

__host__ __device__ float area_sphere(Geom& geom)
{
	return 4.f * PI * geom.scale.x * geom.scale.x; // We're assuming uniform scale
}

__host__ __device__ float area_square(Geom& geom)
{
	return geom.scale.x*geom.scale.y;
}

__host__ __device__ ShadeableIntersection sampleSphere(Vector2f &xi, float& pdf, 
												Vector3f& refPoint, Geom& geom)
{
	ShadeableIntersection it;
	glm::vec4 pObj = glm::vec4(sampling_SquareToSphereUniform(xi), 1.0f); // point in object space
	
	it.intersectPoint = Point3f(geom.transform * pObj);
	Point3f lightCenter = Point3f(geom.transform * glm::vec4(0,0,0,1.0f));
	it.surfaceNormal = glm::normalize(it.intersectPoint - lightCenter);

	float dist = glm::distance(refPoint, lightCenter);
	float sinThetaMax2 = 1 / dist*dist;
	float cosThetaMax = std::sqrt(glm::max((float)0.0f, 1.0f - sinThetaMax2));
	pdf = 1.0f / (2.0f * PI * (1.0f - cosThetaMax));

	return it;
}

__host__ __device__ ShadeableIntersection sampleSquare(Vector2f &xi, float& pdf, Vector3f& refPoint, Geom& geom)
{
	glm::vec4 pObj = glm::vec4(Point3f(xi[0] - 0.5f, xi[1] - 0.5f, 0.0f), 0.0f); // point in object space
	ShadeableIntersection it;

	it.intersectPoint = Point3f(geom.transform * pObj);
	it.surfaceNormal = Point3f(geom.transform * glm::vec4(0, 0, 1, 0.0f));
	
	Vector3f wi = glm::normalize(it.intersectPoint - refPoint);
	float absDot = glm::abs(glm::dot(it.surfaceNormal, -wi));
	float area = geom.scale.x*geom.scale.y;
	float dist = glm::distance(it.intersectPoint, refPoint);
	pdf = dist*dist / (absDot*area);

	return it;
}

__host__ __device__ float sampleShapeArea(Geom& geom, GeomType& lightShape)
{
	float area = 0.0f;
	if (lightShape == SPHERE)
	{
		area = area_sphere(geom);
	}
	else if (lightShape == SQUAREPLANE)
	{
		area = area_square(geom);
	}
	return area;
}

__host__ __device__ ShadeableIntersection sampleShapes(Vector2f &xi, float& pdf, 
													Vector3f& refPoint, Vector3f& refNormal,
													Geom& geom, GeomType& lightShape)
{
	//ONLY works with SPHERES and SQUAREPLANES right now
	ShadeableIntersection isect;

	//if any other shape of light is introduced then you need an if else case here sampling the appropriate case
	if (lightShape == SPHERE)
	{
		isect = sampleSphere(xi, pdf, refPoint, geom);
	}
	else if (lightShape == SQUAREPLANE)
	{
		isect = sampleSquare(xi, pdf, refPoint, geom);
	}

	return isect;
}