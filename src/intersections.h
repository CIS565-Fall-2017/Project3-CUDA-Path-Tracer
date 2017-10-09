#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utilities.h"
#include "utilkern.h"

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
__host__ __device__ float boxIntersectionTest(const Geom& box, const Ray& r,
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
__host__ __device__ float sphereIntersectionTest(const Geom& sphere, const Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    //const float radius = .5;
    const float radius = 1;
	const float rSqrd = radius*radius;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    const float vDotDirection = glm::dot(rt.origin, rt.direction);
	const float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - rSqrd);
    if (radicand < 0) {
        return -1;
    }

    const float squareRoot = sqrt(radicand);
    const float firstTerm = -vDotDirection;
    const float t1 = firstTerm + squareRoot;
    const float t2 = firstTerm - squareRoot;

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

    const glm::vec3 objspaceIntersection = getPointOnRay(rt, t);


    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(glm::mat4(sphere.invTranspose), glm::vec4(objspaceIntersection, 0.f)));


    return glm::length(r.origin - intersectionPoint);
}

//plane is 1x1 centered at 0
__host__ __device__ float planeIntersectionTest(const Geom& plane, const Ray& r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) 
{
	//transform the ray into obj space
	Ray rloc;
	rloc.origin = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));
	rloc.direction = glm::normalize(multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f)));

	glm::vec3 nplane(0, 0, 1);
	//double sided, comment to make single sided
	//nplane.z = glm::dot(-rloc.direction, nplane) ? 1 : -1;
	const float sidelen = 1;
	const float maxextent = sidelen / 2.f;

    //Ray-plane intersection
    const float t = glm::dot(nplane, (glm::vec3(0.5f, 0.5f, 0) - rloc.origin)) / glm::dot(nplane, rloc.direction);
    glm::vec3 ploc = t * rloc.direction + rloc.origin;

    //Check that ploc is within the bounds of the square
    if(t > 0 && ploc.x >= -maxextent && ploc.x <= maxextent 
		&& ploc.y >= -maxextent && ploc.y <= maxextent)
	{
		intersectionPoint = glm::vec3(plane.transform * glm::vec4(ploc, 1));
		normal = glm::normalize(plane.invTranspose * nplane);
		//ComputeTBN(pLocal, &(isect->normalGeometric), &(isect->tangent), &(isect->bitangent));
		//isect->uv = GetUVCoordinates(pLocal);
		return t;
    }
    return -1;
}


__host__ __device__ float shapeIntersectionTest(const Geom& geom, const Ray& ray,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) 
{
	if (geom.type == GeomType::CUBE) {
		return boxIntersectionTest(geom, ray, intersectionPoint, normal, outside);

	} else if (geom.type == GeomType::SPHERE) {
		return sphereIntersectionTest(geom, ray, intersectionPoint, normal,  outside);

	} else if (geom.type == GeomType::PLANE) {
		return planeIntersectionTest(geom, ray, intersectionPoint, normal, outside);
	}
}


////////////////////////////////////////////
//////// FIND CLOSEST INTERSECTION /////////
////////////////////////////////////////////
__host__ __device__ glm::vec3 findClosestIntersection(const Ray& ray,
	const Geom* const geoms, int geoms_size, int& hit_geom_index)
{
	float t;
	float t_min = FLT_MAX;
	hit_geom_index = -1;
	bool outside = true;
	glm::vec3 nisect(0);

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// naive parse through global geoms
	for (int i = 0; i < geoms_size; ++i) {
		const Geom& geom = geoms[i];

		if (geom.type == GeomType::CUBE) {
			t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal,outside);

		} else if (geom.type == GeomType::SPHERE) {
			t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);

		} else if (geom.type == GeomType::PLANE) {
			t = planeIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t) {
			t_min = t;
			hit_geom_index = i;
			nisect = tmp_normal;
		}
	}
	return nisect;
}

////TBN computations
//
//__host__ __device__ glm::vec3 computeTanToObjFromWorldNormalSphere(
//	const Geom& g, const glm::vec3& worldNorm, glm::mat3& tanToObj) {
//
//	const glm::vec3 objNorm = glm::normalize(glm::transpose(g.invTranspose) * worldNorm);
//
//	//N (z)
//	tanToObj[2] = glm::normalize(objNorm);
//	//T (x)
//	tanToObj[0] = glm::normalize(glm::cross(glm::vec3(0, 1, 0), tanToObj[2]));
//	//B (y)
//	tanToObj[1] = glm::normalize(glm::cross(tanToObj[2], tanToObj[0]));
//}
//
//__host__ __device__ glm::vec3 computeTanToObjFromWorldNormalCube(
//	const Geom& g, const glm::vec3& worldNorm, glm::mat3& tanToObj) 
//{
//	const glm::vec3 objNorm = glm::normalize(glm::transpose(g.invTranspose) * worldNorm);
//
//    glm::vec3 objB(1.f,0.f,0.f);
//    if(objNorm.y <= 0.1f) {
//        objB.x = 0.f;
//        objB.y = 1.f;
//    }
//
//	//N (z)
//	tanToObj[2] = objNorm;
//	//B (y)
//	tanToObj[1] = objB;
//	//T (x)
//	tanToObj[0] = glm::normalize(glm::cross(tanToObj[1], tanToObj[2]));
//}
//
//__host__ __device__ glm::vec3 computeTanToObjFromWorldNormalPlane(
//	const Geom& g, const glm::vec3& worldNorm, glm::mat3& tanToObj) 
//{
//	//N (z)
//	tanToObj[2] = glm::vec3(0, 0, 1);
//	//T (x)
//	tanToObj[0] = glm::vec3(1, 0, 0);
//	//B (y)
//	tanToObj[1] = glm::vec3(0, 1, 0);
//}
//
//__host__ __device__ glm::vec3 computeTanToObjFromWorldNormalShape(
//	const Geom& g, const glm::vec3& worldNorm, glm::mat3& tanToObj) {
//
//	if (g.type == GeomType::SPHERE) {
//		computeTanToObjFromWorldNormalSphere(g, worldNorm, tanToObj);
//	} else if( g.type == GeomType::CUBE) {
//		computeTanToObjFromWorldNormalCube(g, worldNorm, tanToObj);
//	} else if( g.type == GeomType::PLANE) {
//		computeTanToObjFromWorldNormalPlane(g, worldNorm, tanToObj);
//	}
//}
