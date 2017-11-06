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
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
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
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
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
    } else if (t1 > 0 && t2 > 0) {
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// CHECKITOUT
/**
* Test intersection between a ray and a transformed square. Untransformed,
* a SquarePlane is assumed to have a side length of 1 and a center of <0,0,0>.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float squareIntersectionTest(Geom square, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) 
{
	Ray rt;
	rt.origin = multiplyMV(square.inverseTransform, glm::vec4(r.origin, 1.0f));
	rt.direction = glm::normalize(multiplyMV(square.inverseTransform, glm::vec4(r.direction, 0.0f)));
	
	float t = glm::dot(glm::vec3(0, 0, 1), (glm::vec3(0.5f, 0.5f, 0) - rt.origin)) / glm::dot(glm::vec3(0, 0, 1), rt.direction);

	glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
	if (t > 0 && objspaceIntersection.x >= -0.5f && objspaceIntersection.x <= 0.5f && 
				 objspaceIntersection.y >= -0.5f && objspaceIntersection.y <= 0.5f)
	{
		intersectionPoint = multiplyMV(square.transform, glm::vec4(objspaceIntersection, 1.f));
		normal = glm::normalize(multiplyMV(square.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
		return glm::length(r.origin - intersectionPoint);
	}

	// Didn't hit
	return -1;
}

__host__ __device__ void computeIntersectionOfRayWithSelectedObject(Ray& ray, Geom& geom,
																ShadeableIntersection& intersection)
{
	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	bool hit_geom = false;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// Naive parse through global geoms
	if (geom.type == CUBE)
	{
		t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
	}
	else if (geom.type == SPHERE)
	{
		t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
	}
	else if (geom.type == SQUAREPLANE)
	{
		t = squareIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
	}

	// Compute the minimum t from the intersection tests to determine what
	// scene geometry object was hit first.
	if (t > 0.0f && t_min > t)
	{
		t_min = t;
		hit_geom = true;
		intersect_point = tmp_intersect;
		normal = tmp_normal;
	}

	if (!hit_geom)
	{
		intersection.t = -1.0f;
	}
	else
	{
		//The ray hits something
		intersection.t = t_min;
		intersection.intersectPoint = intersect_point;
		intersection.materialId = geom.materialid;
		intersection.surfaceNormal = glm::normalize(normal);
	}
}

__host__ __device__ void computeIntersectionsForASingleRay(Ray& ray, ShadeableIntersection& intersection,
														   Geom * geoms, int& geoms_size)
{
	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// Naive parse through global geoms
	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SQUAREPLANE)
		{
			t = squareIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
			intersect_point = tmp_intersect;
			normal = tmp_normal;
		}
	}

	if (hit_geom_index == -1)
	{
		intersection.t = -1.0f;
	}
	else
	{
		//The ray hits something
		intersection.t = t_min;
		intersection.intersectPoint = intersect_point;
		intersection.materialId = geoms[hit_geom_index].materialid;
		intersection.surfaceNormal = glm::normalize(normal);
		intersection.hitGeomIndex = hit_geom_index;
	}
}