#pragma once

#include "sampling.h"
#include "common.h"

__host__ __device__ ShadeableIntersection sampleSphereOld(Vector2f &xi, float& pdf, Geom& geom, float& area)
{
	ShadeableIntersection it;
	glm::vec4 pObj = glm::vec4(sampling_SquareToSphereUniform(xi), 1.0f); // point in object space
	
	it.intersectPoint = Point3f(geom.transform * pObj);
	Point3f lightCenter = Point3f(geom.transform * glm::vec4(0,0,0,1.0f));
	it.surfaceNormal = glm::normalize(it.intersectPoint - lightCenter);

	area = 4.f * PI * geom.scale.x * geom.scale.x; // We're assuming uniform scale
	pdf = 1.0f / area;

	return it;
}

__host__ __device__ ShadeableIntersection sampleSquare(Vector2f &xi, float& pdf, Geom& geom, float& area)
{
	glm::vec4 pObj = glm::vec4(Point3f(xi[0] - 0.5f, xi[1] - 0.5f, 0.0f), 0.0f); // point in object space
	ShadeableIntersection it;

	it.intersectPoint = Point3f(geom.transform * pObj);
	it.surfaceNormal = Point3f(geom.transform * glm::vec4(0, 0, 1, 0.0f));
	
	area = 4.f * PI * geom.scale.x * geom.scale.x; // We're assuming uniform scale
	pdf = 1.0f / area;

	return it;
}

__host__ __device__ ShadeableIntersection sampleSphere(Vector2f &xi, float& pdf, Geom& geom, 
														Vector3f& refPoint, Vector3f& refNormal)
{
	Point3f center = Point3f(geom.transform * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	Vector3f centerToRef = glm::normalize(center - refPoint);
	Vector3f tan, bit;

	CoordinateSystem(centerToRef, &tan, &bit);

	Point3f pOrigin;
	if (glm::dot(center - refPoint, refNormal) > 0)
	{
		pOrigin = refPoint + refNormal * RayEpsilon;
	}
	else
	{
		pOrigin = refPoint - refNormal * RayEpsilon;
	}

	if (glm::distance2(pOrigin, center) <= 1.f) // Radius is 1, so r^2 is also 1
	{
		float area;
		return sampleSphereOld(xi, pdf, geom, area);// Sample(xi, pdf);
	}

	float sinThetaMax2 = 1 / glm::distance2(refPoint, center); // Again, radius is 1
	float cosThetaMax = std::sqrt(glm::max((float)0.0f, 1.0f - sinThetaMax2));
	float cosTheta = (1.0f - xi.x) + xi.x * cosThetaMax;
	float sinTheta = std::sqrt(glm::max((float)0, 1.0f - cosTheta * cosTheta));
	float phi = xi.y * 2.0f * PI;

	float dc = glm::distance(refPoint, center);
	float ds = dc * cosTheta - glm::sqrt(glm::max((float)0.0f, 1 - dc * dc * sinTheta * sinTheta));

	float cosAlpha = (dc * dc + 1 - ds * ds) / (2 * dc * 1);
	float sinAlpha = glm::sqrt(glm::max((float)0.0f, 1.0f - cosAlpha * cosAlpha));

	Vector3f nObj = sinAlpha * glm::cos(phi) * -tan + sinAlpha * glm::sin(phi) * -bit + cosAlpha * -centerToRef;
	Point3f pObj = Point3f(nObj); // Would multiply by radius, but it is always 1 in object space

	ShadeableIntersection isect;

	//    pObj *= radius / glm::length(pObj); // pObj is already in object space with r = 1, so no need to perform this step

	isect.intersectPoint = Point3f(geom.transform * glm::vec4(pObj.x, pObj.y, pObj.z, 1.0f));
	isect.surfaceNormal = Normal3f(geom.inverseTransform * glm::vec4(Normal3f(nObj), 0.0f));

	pdf = 1.0f / (2.0f * PI * (1 - cosThetaMax));

	return isect;
}

__host__ __device__ ShadeableIntersection sampleShapes(Vector2f &xi, float& pdf, 
													Vector3f& refPoint, Vector3f& refNormal,
													Geom& geom, GeomType& lightShape)
{
	//ONLY works with SPHERES and SQUAREPLANES right now
	float area;
	ShadeableIntersection isect;

	//if any other shape of light is introduced then you need an if else case here sampling the appropriate case
	if (lightShape == SPHERE)
	{
		isect = sampleSphere(xi, pdf, geom, refPoint, refNormal);
	}
	else if (lightShape == SQUAREPLANE)
	{
		isect = sampleSquare(xi, pdf, geom, area);
	}

	return isect;
}

//__host__ __device__ ShadeableIntersection sampleShapesOld(Vector2f &xi, float& pdf, Vector3f& refPoint,
//													   Geom& geom, GeomType& lightShape)
//{
//	ONLY works with SPHERES and SQUAREPLANES right now
//	float area;
//	ShadeableIntersection isect;
//
//	if any other shape of light is introduced then you need an if else case here sampling the appropriate case
//	if (lightShape == SPHERE)
//	{
//		isect = sampleSphere(xi, pdf, geom, area);
//	}
//	else if (lightShape == SQUAREPLANE)
//	{
//		isect = sampleSquare(xi, pdf, geom, area);
//	}
//	
//	Vector3f wi = glm::normalize(isect.intersectPoint - refPoint);
//	float absDot = glm::abs(glm::dot(isect.surfaceNormal, -wi));
//
//	if (absDot == 0.0f)
//	{
//		pdf = 0.0f;
//	}
//	else
//	{
//		float dist = glm::distance(isect.intersectPoint, refPoint);
//		pdf = dist*dist / (absDot*area);
//	}
//
//	return isect;
//}