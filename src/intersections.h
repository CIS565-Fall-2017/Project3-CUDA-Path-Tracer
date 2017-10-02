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

__host__ __device__ float meshIntersectionTest(Geom geo, Ray r, Triangle * meshes,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2& uv, glm::vec3&tangent) 
{
	glm::vec3 ro = multiplyMV(geo.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geo.inverseTransform, glm::vec4(r.direction, 0.0f)));
	
	float distance = 10000000.f;
	int triangleIndex = -1;

	for (int i = 0; i < geo.meshData.triangleCount; i++)
	{
		Triangle & triangle = meshes[geo.meshData.offset + i];

		glm::vec3 t = ro - triangle.p1;
		glm::vec3 p = glm::cross(rd, triangle.e2);
		glm::vec3 q = glm::cross(t, triangle.e1);

		float multiplier = 1.f / glm::dot(p, triangle.e1);
		float rayT = multiplier * glm::dot(q, triangle.e2);
		float u = multiplier * glm::dot(p, t);
		float v = multiplier * glm::dot(q, rd);

		if (rayT < distance && rayT >= 0.f && u >= 0.f && v >= 0.f && u + v <= 1.f)
		{
			distance = rayT;
			triangleIndex = i;
			uv = glm::vec2(u, v);
		}
	}

	if (triangleIndex >= 0)
	{
		Ray rt;
		rt.origin = ro;
		rt.direction = rd;

		Triangle & triangle = meshes[triangleIndex];
		outside = true;

		glm::vec3 localP = getPointOnRay(rt, distance);
		glm::vec3 localNormal = triangle.n1 * uv.x + triangle.n2 * uv.y + triangle.n3 * (1.f - uv.x - uv.y);

		intersectionPoint = multiplyMV(geo.transform, glm::vec4(localP, 1.f));
		normal = glm::normalize(multiplyMV(geo.invTranspose, glm::vec4(localNormal, 0.f)));
		//tangent = glm::normalize(multiplyMV(geo.transform, glm::vec4(localTangent, 0.0f)));

		return distance;
	}

	return -1.f;
}
