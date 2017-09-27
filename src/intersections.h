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

__host__ __device__ float TriArea(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3) {
	return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
}

/**
* Returns interpolation of the triangle's three normals based on the point 
* inside the triangle that is given
*/
__host__ __device__ glm::vec3 getNormal(glm::vec3 points[3], glm::vec3 norms[3], glm::vec3 p) {
	float A = TriArea(points[0], points[1], points[2]);
	float A0 = TriArea(points[1], points[2], p);
	float A1 = TriArea(points[0], points[2], p);
	float A2 = TriArea(points[0], points[1], p);
	return glm::normalize(norms[0] * A0 / A + norms[1] * A1 / A + norms[2] * A2 / A);
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
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    //if (!outside) {
    //    normal = -normal;
    //}

    return glm::length(r.origin - intersectionPoint);
}

// TODO
/**
* Test intersection between a ray and a bounding box.
*/
__host__ __device__ bool bbIntersectionTest(Geom bb, Ray r) 
{
	glm::vec3 invDir = glm::vec3(1.f / r.direction.x, 1.f / r.direction.y, 1.f / r.direction.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

	// Check for ray intersection against $x$ and $y$ slabs
	float tMin = (bb.bound[dirIsNeg[0]].x - r.origin.x) * invDir.x;
	float tMax = (bb.bound[1 - dirIsNeg[0]].x - r.origin.x) * invDir.x;
	float tyMin = (bb.bound[dirIsNeg[1]].y - r.origin.y) * invDir.y;
	float tyMax = (bb.bound[1 - dirIsNeg[1]].y - r.origin.y) * invDir.y;

	// Update _tMax_ and _tyMax_ to ensure robust bounds intersection
	float gamma3 = (3 * EPSILON * 0.5) / (1 - 3 * EPSILON * 0.5);
	tMax *= 1 + 2 * gamma3;
	tyMax *= 1 + 2 * gamma3;
	if (tMin > tyMax || tyMin > tMax) return false;
	if (tyMin > tMin) tMin = tyMin;
	if (tyMax < tMax) tMax = tyMax;

	// Check for ray intersection against $z$ slab
	float tzMin = (bb.bound[dirIsNeg[2]].z - r.origin.z) * invDir.z;
	float tzMax = (bb.bound[1 - dirIsNeg[2]].z - r.origin.z) * invDir.z;

	// Update _tzMax_ to ensure robust bounds intersection
	tzMax *= 1 + 2 * gamma3;
	if (tMin > tzMax || tzMin > tMax) return false;
	if (tzMin > tMin) tMin = tzMin;
	if (tzMax < tMax) tMax = tzMax;
	return (tMax > 0);

}

/**
* Test intersection between a ray and a transformed triangle.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float triangleIntersectionTest(Geom tri, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside)
{
	// transform ray
	Ray rt;
	rt.origin = multiplyMV(tri.inverseTransform, glm::vec4(r.origin, 1.0f));
	rt.direction = glm::normalize(multiplyMV(tri.inverseTransform, glm::vec4(r.direction, 0.0f)));

	// ray triangle intersection
	glm::vec3 baryPosition;
	glm::intersectRayTriangle(rt.origin, rt.direction, tri.pos[0], tri.pos[1], tri.pos[2], baryPosition);
	float t = baryPosition.z;

	glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

	intersectionPoint = multiplyMV(tri.transform, glm::vec4(objspaceIntersection, 1.f));
	normal = glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(getNormal(tri.pos, tri.norm, objspaceIntersection), 0.f)));
	outside = glm::dot(normal, r.direction) < 0;

	return t; // negative if outside triangle
}

