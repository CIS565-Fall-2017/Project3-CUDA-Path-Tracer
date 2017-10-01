#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <memory>
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
         glm::vec2 &uv, glm::vec3 &normal, bool &outside) {
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

		glm::vec3 objspaceIntersection = getPointOnRay(q, tmin);

		// UV
		glm::vec3 abs = glm::min(glm::abs(objspaceIntersection), 0.5f);
		glm::vec2 UV(0.0f);//Always offset lower-left corner
		if (abs.x > abs.y && abs.x > abs.z)
		{
			UV = glm::vec2(objspaceIntersection.z + 0.5f, objspaceIntersection.y + 0.5f) / 3.0f;
			//Left face
			if (objspaceIntersection.x < 0)
			{
				UV += glm::vec2(0, 0.333f);
			}
			else
			{
				UV += glm::vec2(0, 0.667f);
			}
		}
		else if (abs.y > abs.x && abs.y > abs.z)
		{
			UV = glm::vec2(objspaceIntersection.x + 0.5f, objspaceIntersection.z + 0.5f) / 3.0f;
			//Left face
			if (objspaceIntersection.y < 0)
			{
				UV += glm::vec2(0.333f, 0.333f);
			}
			else
			{
				UV += glm::vec2(0.333f, 0.667f);
			}
		}
		else
		{
			UV = glm::vec2(objspaceIntersection.x + 0.5f, objspaceIntersection.y + 0.5f) / 3.0f;
			//Left face
			if (objspaceIntersection.z < 0)
			{
				UV += glm::vec2(0.667f, 0.333f);
			}
			else
			{
				UV += glm::vec2(0.667f, 0.667f);
			}
		}
		uv = UV;


		glm::vec3 intersectionPoint = multiplyMV(box.transform, glm::vec4(objspaceIntersection, 1.0f));
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
         glm::vec2 &uv, glm::vec3 &normal, bool &outside) {
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

	// UV
	glm::vec3 p = glm::normalize(objspaceIntersection);
	float phi = atan2f(p.z, p.x);
	if (phi < 0)
	{
		phi += TWO_PI;
	}
	float theta = glm::acos(p.y);
	uv = glm::vec2(1.0f - phi / TWO_PI, 1 - theta / PI);

	glm::vec3 intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
	//since we want to judge ray goes in or out of sphere,
	//Don't want our normal reversed and it always points outside
    //if (!outside) {
    //    normal = -normal;
    //}

    return glm::length(r.origin - intersectionPoint);
}

/// Float approximate-equality comparison
//template<typename T>
//__host__ __device__ inline bool fequal(T a, T b, T epsilon = 0.0001) {
//	if (a == b) {
//		// Shortcut
//		return true;
//	}
//
//	const T diff = std::abs(a - b);
//	if (a * b == 0) {
//		// a or b or both are zero; relative error is not meaningful here
//		return diff < (epsilon * epsilon);
//	}
//
//	return diff / (std::abs(a) + std::abs(b)) < epsilon;
//}
//
//__host__ __device__ bool TriangleIntersect(const Ray& r, const Triangle& tri, glm::vec3& baryPosition)
//{
//	glm::vec3 planeNormal = tri.normals[0];
//	glm::vec3 points[3];
//	points[0] = tri.vertices[0];
//	points[1] = tri.vertices[1];
//	points[2] = tri.vertices[2];
//
//
//	//1. Ray-plane intersection
//	float t = glm::dot(planeNormal, (points[0] - r.origin)) / glm::dot(planeNormal, r.direction);
//	if (t < 0.f) return false;
//
//	glm::vec3 P = r.origin + t * r.direction;
//	//2. Barycentric test
//	float S = 0.5f * glm::length(glm::cross(points[0] - points[1], points[0] - points[2]));
//	float s1 = 0.5f * glm::length(glm::cross(P - points[1], P - points[2])) / S;
//	float s2 = 0.5f * glm::length(glm::cross(P - points[2], P - points[0])) / S;
//	float s3 = 0.5f * glm::length(glm::cross(P - points[0], P - points[1])) / S;
//	float sum = s1 + s2 + s3;
//
//	if (s1 >= 0.f && s1 <= 1.f && s2 >= 0.f && s2 <= 1.f && s3 >= 0.f && s3 <= 1.f && fequal(sum, 1.0f)) {
//		baryPosition.x = s1 / S;
//		baryPosition.y = s2 / S;
//		baryPosition.z = t;
//		return true;
//	}
//	return false;
//}

__host__ __device__ float meshIntersectionTest(Geom mesh, Triangle* tris, 
	Ray r, glm::vec2 &uv, glm::vec3 &normal, bool &outside) {

	float tMin = FLT_MAX;
	int nearestTriIndex = -1;
	glm::vec3 baryPosition(0.f);
	glm::vec3 minBaryPosition(0.f);

	int endIndex = mesh.meshTriangleEndIdx;


	for(int i = mesh.meshTriangleStartIdx; i < endIndex; i++)
	{	
		// should be counter-clock wise
		if (glm::intersectRayTriangle(r.origin, r.direction, 
								      tris[i].vertices[0], tris[i].vertices[1], tris[i].vertices[2],
									  baryPosition)) {
			// Only consider triangls in the ray direction
			// no triangels back!
			if (baryPosition.z > 0.f && baryPosition.z < tMin) {
				tMin = baryPosition.z;
				minBaryPosition = baryPosition;
				nearestTriIndex = i;
			}
		}

		//if (TriangleIntersect(r_temp, thisTriangle, baryPosition)) {
		//	if (baryPosition.z > 0.f && baryPosition.z < tMin) {
		//		tMin = baryPosition.z;
		//		minBaryPosition = baryPosition;
		//		nearestTriIndex = i;
		//	}
		//}
	}
	
	if (nearestTriIndex == -1) {
		return -1;
	}

	// set uv and normal
	Triangle nearestIntersectTri = tris[nearestTriIndex];

	uv = nearestIntersectTri.uvs[0] * minBaryPosition.x +
		nearestIntersectTri.uvs[1] * minBaryPosition.y +
		nearestIntersectTri.uvs[2] * (1.0f - minBaryPosition.x - minBaryPosition.y);

	normal = nearestIntersectTri.normals[0] * minBaryPosition.x +
			nearestIntersectTri.normals[1] * minBaryPosition.y +
			nearestIntersectTri.normals[2] * (1.0f - minBaryPosition.x - minBaryPosition.y);

	return tMin;
}



__host__ __device__ inline float AbsDot(const glm::vec3 &a, const glm::vec3 &b)
{
	return glm::abs(glm::dot(a, b));
}

__host__ __device__ inline glm::vec3 Reflect(const glm::vec3 &wo, const glm::vec3 &n) {
	return -wo + 2.0f * glm::dot(wo, n) * n;
}

__host__ __device__ inline glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
	return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__ inline bool Refract(const glm::vec3 &wi, const glm::vec3 &n, float eta,
	glm::vec3 *wt) {
	// Compute cos theta using Snell's law
	float cosThetaI = glm::dot(n, wi);
	//float sin2ThetaI = std::max(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaI = fmaxf (0.f, (1.0f - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = std::sqrt(1 - sin2ThetaT);
	*wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
	return true;
}

#ifdef ENABLE_BVH
// Control the search depth of BVH by controlling interior nodes number need to be search 
#define MAX_BVH_INTERIOR_LEVEL 64 // assume the maximum bvh interior node level is 64
#endif 


// Mainly target to Call in GPU
__host__ __device__ bool IntersectBVH(const Ray &ray, ShadeableIntersection * isect,
	const LinearBVHNode *bvh_nodes,
	const Triangle* primitives)
{
	if (!bvh_nodes) return false;

	bool hit = false;

	//   bool SameLevelBoundingBoxIntersect = false;

	glm::vec3 invDir(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
	int dirIsNeg[3] = { invDir.x < 0.f, invDir.y < 0.f, invDir.z < 0.f };


	// Follow ray through BVH nodes to find primitive intersections
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisit[MAX_BVH_INTERIOR_LEVEL]; 

	while (true) {

		const LinearBVHNode *node = &bvh_nodes[currentNodeIndex];
		// Check ray against BVH node
		float temp_t = 0.f;
		// For the very first bounding box, we don't know whether the ray
		// interects (root node) or not, so wo do initial test here
		// but for the following bounding box added, we
		// make sure they are SURE to intersect!
		bool node_isect = (currentNodeIndex == 0) ? node->bounds.Intersect(ray, &temp_t) : true;

		if (node->bounds.IntersectP(ray, invDir, dirIsNeg)) {

			// If this is a leaf node
			if (node->nPrimitives > 0) {

				// Intersect ray with EVERY primitive in leaf BVH node
				for (int i = 0; i < node->nPrimitives; i++) {
					ShadeableIntersection temp_isect;
					if (primitives[node->primitivesOffset + i].Intersect(ray, &temp_isect)) {

						hit = true;

						// if iscet is still null,
						// we need initialize it
						if (isect->t == -1.0f) {
							(*isect) = temp_isect;
						}
						else {
							if (temp_isect.t < isect->t) {
								(*isect) = temp_isect;
							}
						}
					}
				}

				// If it hits one primitive, loop break;
				// since bounding box is put in ToVisit list
				// according to t value(from small to large)

				//    if(toVisitOffset == 0 || hit) break;

				// no.. actually, it's incorrect!
				// can't stop loop, there may be intersection with smaller t value!
				if (toVisitOffset == 0) break;


				//                if(SameLevelBoundingBoxIntersect){
				//                    if(toVisitOffset == 0) break;
				//                }
				//                else{
				//                    if(toVisitOffset == 0 || hit) break;
				//                }
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}

			// If this is a interior node
			else {
				// ----------- Depth control ---------------
				// if toVisitOffset reaches maximum
				// we don't want add more index to nodesToVisit Array
				// we just give up this interior node and handle previous nodes instead 
				if (toVisitOffset == MAX_BVH_INTERIOR_LEVEL) {
					currentNodeIndex = nodesToVisit[--toVisitOffset];
					continue;
				}
				// -----------------------------------------
				
				// add index to nodes to visit
				if (dirIsNeg[node->axis]) {
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node->secondChildOffset;
				}
				else {
					nodesToVisit[toVisitOffset++] = node->secondChildOffset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}

		// If the root node hits nothing
		else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}

	}

	return hit;
}