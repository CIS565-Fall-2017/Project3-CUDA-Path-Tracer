#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
	return r.origin + glm::normalize(r.direction) * t;
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
	return glm::vec3(m * v);
}

__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2 &uv, glm::vec3 & tangent) {
	Ray q;
	q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
			float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) {
				tmin = ta;
				tmin_n = n;
			}
			if (tb < tmax) {
				tmax = tb;
				tmax_n = n;
			}
		}
	}

	if (tmax >= tmin && tmax > 0) {
		outside = true;
		if (tmin <= 0) {
			tmin = tmax;
			tmin_n = tmax_n;
			outside = false;
		}
		glm::vec3 localP = q.origin + q.direction * tmin;

		// Triplanar
		uv = glm::vec2(localP.z, localP.y) * tmin_n.x;
		uv += glm::vec2(localP.x, localP.z) * tmin_n.y;
		uv += glm::vec2(localP.x, localP.y) * tmin_n.z;

		// [-.5,.5] -> [0,1]
		uv += glm::vec2(.5f);

		glm::vec3 localTangent;
		if (abs(tmin_n.x) < SQRT_OF_ONE_THIRD)
			localTangent = glm::vec3(1, 0, 0);
		else if (abs(tmin_n.y) < SQRT_OF_ONE_THIRD)
			localTangent = glm::vec3(0, 1, 0);
		else 
			localTangent = glm::vec3(0, 0, 1);

		localTangent = glm::normalize(glm::cross(tmin_n, localTangent));

		intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
		tangent = glm::normalize(multiplyMV(box.transform, glm::vec4(localTangent, 0.0f)));
		return glm::length(r.origin - intersectionPoint);
	}
	return -1;
}

__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2& uv, glm::vec3&tangent) {
	float radius = .5;

	glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	float vDotDirection = glm::dot(rt.origin, rt.direction);
	float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
	if (radicand < 0) {
		return -1;
	}

	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;

	float t = 0;
	if (t1 < 0 && t2 < 0) {
		return -1;
	}
	else if (t1 > 0 && t2 > 0) {
		t = min(t1, t2);
		outside = true;
	}
	else {
		t = max(t1, t2);
		outside = false;
	}

	glm::vec3 localP = getPointOnRay(rt, t);

	glm::vec3 localTangent;
	if (abs(localP.x) < SQRT_OF_ONE_THIRD)
		localTangent = glm::vec3(1, 0, 0);
	else if (abs(localP.y) < SQRT_OF_ONE_THIRD)
		localTangent = glm::vec3(0, 1, 0);
	else
		localTangent = glm::vec3(0, 0, 1);

	localTangent = glm::normalize(glm::cross(localP, localTangent));

	intersectionPoint = multiplyMV(sphere.transform, glm::vec4(localP, 1.f));
	normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(localP, 0.f)));
	tangent = glm::normalize(multiplyMV(sphere.transform, glm::vec4(localTangent, 0.0f)));

	float phi = glm::atan(localP.z, localP.x);
	if (phi < 0)
		phi += glm::two_pi<float>();

	uv = glm::vec2(1.f - (phi / glm::two_pi<float>()), 1.f - (glm::acos(localP.y) / glm::pi<float>()));
	if (!outside) {
		//normal = -normal;
	}

	return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool aabbIntersectionTest(Ray r, glm::vec3 min, glm::vec3 max, float & tmin, float & tmax) {

	tmin = -1e38f;
	tmax = 1e38f;

	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = r.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (min[xyz] - r.origin[xyz]) / qdxyz;
			float t2 = (max[xyz] - r.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			if (ta > 0 && ta > tmin)
				tmin = ta;
			if (tb < tmax)
				tmax = tb;
		}
	}

	if (tmax >= tmin && tmax > 0)
	{
		/*if (tmin <= 0)
			tmin = tmax;*/

		return true;
	}

	return false;
}

__host__ __device__ float meshIntersectionTest(Geom geo, Ray r, void * meshes,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2& uv, glm::vec3&tangent) 
{
	outside = true;
	float minDistance = -1000000.f;
	float maxDistance = 1000000.f;

	// If we dont hit the AABB, return!
	if (!aabbIntersectionTest(r, geo.meshData.minAABB, geo.meshData.maxAABB, minDistance, maxDistance))
		return -1.f;

	glm::vec3 invRayDir = glm::vec3(1.f / r.direction.x, 1.f / r.direction.y, 1.f / r.direction.z);
	StackData stack[64] = { 0 };
	int stackTop = 0;

	bool hit = false;
	int currentNode = 0;
	int triangleSize = 10;

	float intersectionDistance = 1000000.f;
	int * compactNodes = (int*)(meshes) + geo.meshData.offset;

	// The stack approach is very similar to pbrtv3
	while (currentNode != -1 && stackTop < 64)
	{
		// If on a previous loop there was an intersection and is closer
		// than the current node min distance, don't even start checking intersections
		if (intersectionDistance < minDistance)
			break;

		CompactNode * cNode = (CompactNode*)(compactNodes + currentNode);
		int leftNode = cNode->leftNode;
		int rightNode = cNode->rightNode;
		float split = cNode->split;
		int axis = cNode->axis;

		// Leaf
		if (leftNode == -1 && rightNode == -1)
		{
			int primitiveCount = cNode->primitiveCount;
			CompactTriangle * flatElements = (CompactTriangle*)(compactNodes + currentNode + 5);

			// Check intersection with all primitives inside this node
			for (int i = 0; i < primitiveCount; i++)
			{
				CompactTriangle& tri = flatElements[i];

				glm::vec3 e1(tri.e1x, tri.e1y, tri.e1z);
				glm::vec3 e2(tri.e2x, tri.e2y, tri.e2z);
				glm::vec3 p1(tri.p1x, tri.p1y, tri.p1z);

				glm::vec3 t = r.origin - p1;
				glm::vec3 p = glm::cross(r.direction, e2);
				glm::vec3 q = glm::cross(t, e1);

				float multiplier = 1.f / glm::dot(p, e1);
				float rayT = multiplier * glm::dot(q, e2);
				float u = multiplier * glm::dot(p, t);
				float v = multiplier * glm::dot(q, r.direction);

				if (rayT < intersectionDistance && rayT >= 0.f && u >= 0.f && v >= 0.f && u + v <= 1.f)
				{
					intersectionDistance = rayT;
					//fint->elementIndex = (i * triangleSize) + currentNode + 5;

					glm::vec3 n1(tri.n1x, tri.n1y, tri.n1z);
					glm::vec3 n2(tri.n2x, tri.n2y, tri.n2z);
					glm::vec3 n3(tri.n3x, tri.n3y, tri.n3z);

					glm::vec3 localP = getPointOnRay(r, intersectionDistance);
					glm::vec3 localNormal = glm::normalize(n1 * uv.x + n2 * uv.y + n3 * (1.f - uv.x - uv.y));

					intersectionPoint = multiplyMV(geo.transform, glm::vec4(localP, 1.f));
					normal = glm::normalize(multiplyMV(geo.invTranspose, glm::vec4(localNormal, 0.f)));
					//tangent = glm::normalize(multiplyMV(geo.transform, glm::vec4(localTangent, 0.0f)));

					hit = true;
				}
			}

			if (stackTop > 0)
			{
				stackTop--;
				currentNode = stack[stackTop].nodeOffset;
				minDistance = stack[stackTop].minDistance;
				maxDistance = stack[stackTop].maxDistance;
			}
			else break; // There's no other object in the stack, we finished iterating!
		}
		else
		{
			float t = (split - r.origin[axis]) * invRayDir[axis];
			int nearNode = leftNode;
			int farNode = rightNode;

			if (r.origin[axis] >= split && !(r.origin[axis] == split && r.direction[axis] < 0))
			{
				nearNode = rightNode;
				farNode = leftNode;
			}

			if (t > maxDistance || t <= 0)
				currentNode = nearNode;
			else if (t < minDistance)
				currentNode = farNode;
			else
			{
				stack[stackTop].nodeOffset = farNode;
				stack[stackTop].minDistance = t;
				stack[stackTop].maxDistance = maxDistance;
				stackTop++; // Increment the stack
				currentNode = nearNode;
				maxDistance = t;
			}
		}
	}

	if (hit)
		return intersectionDistance;

//	return hit;
//}


	//glm::vec3 ro = multiplyMV(geo.inverseTransform, glm::vec4(r.origin, 1.0f));
	//glm::vec3 rd = glm::normalize(multiplyMV(geo.inverseTransform, glm::vec4(r.direction, 0.0f)));
	//
	//float distance = 10000000.f;
	//int triangleIndex = -1;

	//for (int i = 0; i < geo.meshData.triangleCount; i++)
	//{
	//	Triangle triangle = ((Triangle*)meshes)[geo.meshData.offset + i];

	//	glm::vec3 t = ro - triangle.p1;
	//	glm::vec3 p = glm::cross(rd, triangle.e2);
	//	glm::vec3 q = glm::cross(t, triangle.e1);

	//	float multiplier = 1.f / glm::dot(p, triangle.e1);
	//	float rayT = multiplier * glm::dot(q, triangle.e2);
	//	float u = multiplier * glm::dot(p, t);
	//	float v = multiplier * glm::dot(q, rd);

	//	if (rayT < distance && rayT >= 0.f && u >= 0.f && v >= 0.f && u + v <= 1.f)
	//	{
	//		distance = rayT;
	//		triangleIndex = i;
	//		uv = glm::vec2(u, v);
	//	}
	//}

	//if (triangleIndex >= 0)
	//{
	//	Ray rt;
	//	rt.origin = ro;
	//	rt.direction = rd;

	//	Triangle & triangle = ((Triangle*)meshes)[triangleIndex];
	//	outside = true;

	//	glm::vec3 localP = getPointOnRay(rt, distance);
	//	glm::vec3 localNormal = triangle.n1 * uv.x + triangle.n2 * uv.y + triangle.n3 * (1.f - uv.x - uv.y);

	//	intersectionPoint = multiplyMV(geo.transform, glm::vec4(localP, 1.f));
	//	normal = glm::normalize(multiplyMV(geo.invTranspose, glm::vec4(localNormal, 0.f)));
	//	//tangent = glm::normalize(multiplyMV(geo.transform, glm::vec4(localTangent, 0.0f)));

	//	return distance;
	//}

	return -1.f;
}
