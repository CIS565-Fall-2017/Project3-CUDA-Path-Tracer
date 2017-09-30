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
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
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

__host__ __device__ bool aabbBoxIntersect(const Ray& r, glm::vec3 min, glm::vec3 max)
{
	float tnear = FLT_MIN;
	float tfar = FLT_MAX;

	for (int i = 0; i<3; i++) 
	{
		float t0, t1;

		if (fabs(r.direction[i]) < EPSILON)
		{
			if (r.origin[i] < min[i] || r.origin[i] > max[i])
				return false;
			else
			{
				t0 = FLT_MIN;
				t1 = FLT_MAX;
			}
		}
		else
		{
			t0 = (min[i] - r.origin[i]) / r.direction[i];
			t1 = (max[i] - r.origin[i]) / r.direction[i];
		}

		tnear = glm::max(tnear, glm::min(t0, t1));
		tfar = glm::min(tfar, glm::max(t0, t1));
	}

	if (tfar < tnear) return false; // no intersection

	if (tfar < 0) return false; // behind origin of ray

	return true;

}

__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, Vertex *vertices) {

	//world to local
	Ray rt;
	rt.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	rt.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	if (!aabbBoxIntersect(rt, mesh.bbox_min, mesh.bbox_max))
		return -1;

	int start_index = mesh.start_Index;
	int num_vertices = mesh.vertices_Num;

	float t = FLT_MAX;

	for (int i = 0; i < num_vertices; i += 3) {
		glm::vec3 baryPosition;
		glm::vec3 v0, v1, v2;
		v0 = vertices[start_index + i].position;
		v1 = vertices[start_index + i + 1].position;
		v2 = vertices[start_index + i + 2].position;

		bool res = glm::intersectRayTriangle(rt.origin, rt.direction, v0, v1, v2, baryPosition);
		float ti = FLT_MAX;
		glm::vec3 intersectPoint = rt.origin + rt.direction * baryPosition.z;
		if (res)
			ti = glm::length(intersectPoint - rt.origin) / glm::length(rt.direction);
		if (ti < t) {
			t = ti;
			normal = glm::normalize(vertices[start_index + i].normal + vertices[start_index + i + 1].normal + vertices[start_index + i + 2].normal);
			intersectionPoint = intersectPoint;
		}
	}

	// no intersection
	if (t == FLT_MAX) {
		return -1;
	}

	int sign = 1;
	if (glm::dot(rt.direction, normal) >= 0) {
		outside = false;
	//	sign = -1;
	}
	else {
	//	sign = 1;
		outside = true;
		//printf("outside\n");
	}

	

	//local to world
	intersectionPoint = glm::vec3(multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.0f)));
	normal = float(sign) * glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normal, 0.0f)));
	//printf ("Hit Triangle: %f %f %f \n", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);

	
	return glm::length(r.origin - intersectionPoint);
}