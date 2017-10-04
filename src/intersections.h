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
__host__ __device__ float boxIntersectionTest(const Geom& box, Ray r,
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

	return glm::length(r.origin - intersectionPoint) > EPSILON ? glm::length(r.origin - intersectionPoint) : -1.f;
}


/**
* Test intersection between a ray and a transformed sphere. Untransformed,
* the sphere always has radius 0.5 and is centered at the origin.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float planeIntersectionTest(Geom plane, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal) {
	
	Ray r_loc;
	r_loc.origin = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));
	r_loc.direction = multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f));

	float t = glm::dot(glm::vec3(0, 0, 1), (glm::vec3(0.5f, 0.5f, 0) - r_loc.origin)) / glm::dot(glm::vec3(0, 0, 1), r_loc.direction);
	glm::vec3 p = glm::vec3(t * r_loc.direction + r_loc.origin);

	if (t > 0 && p.x >= -0.5f && p.x <= 0.5f && p.y >= -0.5f && p.y <= 0.5f) {
		intersectionPoint = multiplyMV(plane.transform, glm::vec4(p,1));
		normal = glm::normalize(multiplyMV(plane.invTranspose, glm::vec4(0, 0, 1, 0)));
		return t;
	}

	return -1;
}


/***********************************************

MESH INTERSECTION ZONE
ENTER WITH CAUTION

************************************************/

//Gets Area of a defined Triangle
__host__ __device__ float TriArea(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3)
{
	return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
}

//Does Barycentric Interpolation Between Triangle Points and reference point

__host__ __device__ glm::vec3 getNormal(const Geom* tri, const glm::vec3& p) {
	glm::vec3 p0 = tri->positions[0];
	glm::vec3 p1 = tri->positions[1];
	glm::vec3 p2 = tri->positions[2];

	glm::vec3 n0 = tri->normals[0];
	glm::vec3 n1 = tri->normals[1];
	glm::vec3 n2 = tri->normals[2];

	float A = TriArea(p0, p1, p2);
	float A0 = TriArea(p1, p2, p);
	float A1 = TriArea(p0, p2, p);
	float A2 = TriArea(p0, p1, p);

	return glm::normalize(n0 * A0 / A + n1 * A1 / A + n2 * A2 / A);
}

/**
* Test intersection between a ray and a transformed mesh. Untransformed,
* the mesh is cool.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
//For some reason 561 code didn't work so I used:
//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__host__ __device__ float triangleIntersectionTest(const Geom& tri, Ray r_world,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool& outside) {

	Ray r;
	r.origin = multiplyMV(tri.inverseTransform, glm::vec4(r_world.origin, 1.0f));
	r.direction = glm::normalize(multiplyMV(tri.inverseTransform, glm::vec4(r_world.direction, 0.0f)));

	glm::vec3 vertex0 = tri.positions[0];
	glm::vec3 vertex1 = tri.positions[1];
	glm::vec3 vertex2 = tri.positions[2];
	glm::vec3 edge1, edge2, h, s, q;
	float a, f, u, v;

	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;

	h = glm::cross(r.direction, edge2);
	a = glm::dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return -1;
	f = 1.f / a;
	s = r.origin - vertex0;
	u = f * (glm::dot(s,h));
	if (u < EPSILON || u > 1.0 - EPSILON) return -1;

	q = glm::cross(s,edge1);
	v = f * glm::dot(r.direction,q);
	if (v < EPSILON || u + v > 1.0 - EPSILON) return -1;
	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * glm::dot(edge2,q);

	//Local Normal:
	glm::vec3 norm = glm::normalize(glm::cross(edge1, edge2));

	if (t > EPSILON) // ray intersection
	{
		intersectionPoint = multiplyMV(tri.transform, glm::vec4(getPointOnRay(r,t),1.f));
		normal = getNormal(&tri, intersectionPoint);
		outside = glm::dot(norm, r.direction) < EPSILON;
		return glm::length(r_world.origin - intersectionPoint);
	}
	else // This means that there is a line intersection but not a ray intersection.
		return -1;

}


/**
* Test intersection between a ray and a transformed mesh. Untransformed,
* the mesh is cool.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
/**
__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal) {

	Ray r_loc;
	r_loc.origin    = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	r_loc.direction = multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f));

	float closest_t = -1;
	Triangle closestTri;

	//For every triangle:
	for (int i = 0; i < mesh.tri_count; i++) {
		int increment = (mesh.tri_index + i);
		const Triangle tri = dev_triangles[increment];
		float tri_t = triangleIntersectionTest(tri, r_loc, intersectionPoint, normal);
		if (tri_t > 0 && (tri_t < closest_t || closest_t < 0)) {
			closest_t = tri_t;
			closestTri = tri;
		}
	}

	if (closest_t > 0)
	{
		glm::vec3 p = glm::vec3(closest_t * r_loc.direction + r_loc.origin);
		glm::vec3 n = getNormal(&closestTri, p);
		return closest_t;
	}

	return -1;
}
*/

