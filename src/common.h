#pragma once

#include "sampleShapes.h"

#define COMPEPSILON 0.000000001f
#define RayEpsilon 0.000005f

__host__ __device__ bool fequals(float f1, float f2)
{
	if ((glm::abs(f1 - f2) < COMPEPSILON))
	{
		return true;
	}
	return false;
}

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

// Create a set of axes to form the basis of a coordinate system
// given a single vector v1.
__host__ __device__ void CoordinateSystem(const Vector3f& v1, Vector3f* v2, Vector3f* v3)
{
	if (std::abs(v1.x) > std::abs(v1.y))
	{
		*v2 = Vector3f(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
	}
	else
	{
		*v2 = Vector3f(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
	}
	*v3 = glm::cross(v1, *v2);
}

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