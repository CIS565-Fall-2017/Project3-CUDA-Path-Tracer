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
    //return r.origin + (t - .0001f) * glm::normalize(r.direction);
	return r.origin + (t - .0001f) * r.direction;
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
    //q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
	q.direction = multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f));

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
        //return glm::length(r.origin - intersectionPoint);
		return tmin;
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

    Ray rt;
    rt.origin = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    //rt.direction = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
	rt.direction = multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f));

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
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
        outside = false;
    }

    const glm::vec3 objspaceIntersection = getPointOnRay(rt, t);


    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(glm::mat4(sphere.invTranspose), glm::vec4(objspaceIntersection, 0.f)));


    //return glm::length(r.origin - intersectionPoint);
	return t;
}

//plane is 1x1 centered at 0
__host__ __device__ float planeIntersectionTest(const Geom& plane, const Ray& r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) 
{
	//transform the ray into obj space
	Ray rloc;
	rloc.origin = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));
	//rloc.direction = glm::normalize(multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f)));
	rloc.direction = multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f));

	glm::vec3 nplane(0, 0, 1);
	//double sided, comment to make single sided
	//nplane.z = glm::dot(-rloc.direction, nplane) ? 1 : -1;
	const float sidelen = 1;
	const float maxextent = sidelen / 2.f;

    //Ray-plane intersection
    const float t = glm::dot(nplane, (glm::vec3(0.5f, 0.5f, 0) - rloc.origin)) / glm::dot(nplane, rloc.direction);
	glm::vec3 ploc = getPointOnRay(rloc, t);

    //Check that ploc is within the bounds of the square
    if(t > 0 && ploc.x >= -maxextent && ploc.x <= maxextent 
		&& ploc.y >= -maxextent && ploc.y <= maxextent)
	{
		intersectionPoint = glm::vec3(plane.transform * glm::vec4(ploc, 1));
		normal = glm::normalize(plane.invTranspose * nplane);
		//ComputeTBN(pLocal, &(isect->normalGeometric), &(isect->tangent), &(isect->bitangent));
		//isect->uv = GetUVCoordinates(pLocal);
		//return glm::length(r.origin - intersectionPoint);
		return t;
    }
    return -1;
}
__host__ __device__ float isectAABB(const AABB& b, const Ray& r, const glm::vec3& dir_inv) {
    float t1 = (b.min[0] - r.origin[0])*dir_inv[0];
    float t2 = (b.max[0] - r.origin[0])*dir_inv[0];
 
    float tmin = glm::min(t1, t2);
    float tmax = glm::max(t1, t2);
 
    for (int i = 1; i < 3; ++i) {
        t1 = (b.min[i] - r.origin[i])*dir_inv[i];
        t2 = (b.max[i] - r.origin[i])*dir_inv[i];
 
        tmin = glm::max(tmin, glm::min(t1, t2));
        tmax = glm::min(tmax, glm::max(t1, t2));
    }
 
	if (tmax > glm::max(tmin, 0.f)) {
		return (tmin < 0.f) ? tmax : tmin;
	} else {
		return -1.f;
	}

////bool Bounds3f::Intersect(const Ray &r, float* t) const {
//    //TODO
//	float t;
//    float t_n = -FLT_MAX;
//    float t_f = FLT_MAX;
//    const glm::vec2 minmax_xyz[3] = {glm::vec2(b.min.x, b.max.x), glm::vec2(b.min.y, b.max.y), glm::vec2(b.min.z, b.max.z)};
//    for(int i = 0; i < 3; i++){
//        //Ray parallel to slab check
//        const float minbound = minmax_xyz[i][0];
//        const float maxbound = minmax_xyz[i][1];
//        if(r.direction[i] == 0){
//            if(r.origin[i] < minbound || r.origin[i] > maxbound){
//				return -1.f;
//            }
//        }
//        //If not parallel, do slab intersect check
//        float t0 = (minbound - r.origin[i])/r.direction[i];
//        float t1 = (maxbound - r.origin[i])/r.direction[i];
//        if(t0 > t1) {
//            float temp = t1;
//            t1 = t0;
//            t0 = temp;
//        }
//        if(t0 > t_n) {
//            t_n = t0;
//        }
//        if(t1 < t_f) {
//            t_f = t1;
//        }
//    }
//    if(t_n < t_f) {
//        if(r.origin.x >= minmax_xyz[0][0] && r.origin.y >= minmax_xyz[1][0] && r.origin.z >= minmax_xyz[2][0]
//        && r.origin.x <= minmax_xyz[0][1] && r.origin.y <= minmax_xyz[1][1] && r.origin.z <= minmax_xyz[2][1]){
//			return t_n;
//        } else {
//            float temp_t = t_n > 0 ? t_n : t_f;
//            if(temp_t < 0) {
//				return -1.f;
//            }
//            return temp_t;
//        }
//    } else {//If t_near was greater than t_far, we did not hit the cube
//        return -1.f;
//    }
////}

}

__host__ __device__ void interpTriangleValues(const glm::ivec3& indices,
	const Vertex* dev_TriVertices, const float* triAreas, glm::vec3& normal) 
{
	//cpu version has tbn calculation using uv's, this was already done and stored in the vertex data by assimp
	const glm::vec3 normals[3] = { dev_TriVertices[indices[0]].nor, dev_TriVertices[indices[1]].nor, dev_TriVertices[indices[2]].nor };
	const glm::vec2 uvs[3] = { dev_TriVertices[indices[0]].uv, dev_TriVertices[indices[1]].uv, dev_TriVertices[indices[2]].uv };

	//TODO: might have to involve tan and its w value to figure out the proper normal

	//triAreas[0] hold the invers of the total area. see isectTriangle for 
	//which entries correspond to which barycentric internal triangle(should be opposite of its corresponding interp vertex
	normal = glm::vec3(glm::normalize( normals[0] * triAreas[1] * triAreas[0] +
									   normals[1] * triAreas[2] * triAreas[0] +
									   normals[2] * triAreas[3] * triAreas[0]   ));
}

__host__ __device__ float isectTriangle(const glm::ivec3& indices, 
	const Vertex* dev_TriVertices, const Ray& rloc, float* triAreas) 
{
	//0. gather triangles points and calculate the planeNormal
	const glm::vec3 points[3] = {dev_TriVertices[indices[0]].pos, dev_TriVertices[indices[1]].pos, dev_TriVertices[indices[2]].pos };
	const glm::vec3 edgeCross = glm::cross(points[1] - points[0], points[2] - points[0]);//how does assimp wind the verts? ccw? cw?
	const glm::vec3 planeNormal = glm::normalize(edgeCross);

    //1. Ray-plane intersection
    const float t =  glm::dot(planeNormal, (points[0] - rloc.origin)) / glm::dot(planeNormal, rloc.direction);
    if(t < 0) return -1.f;

    const glm::vec3 ploc = rloc.origin + t * rloc.direction;

    //2. Barycentric test
    triAreas[0] = 0.5f * glm::length(edgeCross);
	triAreas[1] = 0.5f * glm::length(glm::cross(ploc - points[1], ploc - points[2])) / triAreas[0];
	triAreas[2] = 0.5f * glm::length(glm::cross(ploc - points[2], ploc - points[0])) / triAreas[0];
    triAreas[3] = 0.5f * glm::length(glm::cross(ploc - points[0], ploc - points[1])) / triAreas[0];
	const float sum = triAreas[1] + triAreas[2] + triAreas[3];

	return (triAreas[1] >= 0 && triAreas[1] <= 1 && 
			triAreas[2] >= 0 && triAreas[2] <= 1 &&
			triAreas[3] >= 0 && triAreas[3] <= 1 && 
			fequal(sum, 1.0f)) ? 
			t : -1.f;
}


__host__ __device__ float modelIntersectionTest(const Geom& model, const Ray& r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, 
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices) 
{
	//transform the ray into 
    Ray rloc;
    rloc.origin = multiplyMV(model.inverseTransform, glm::vec4(r.origin, 1.0f));
	rloc.direction = multiplyMV(model.inverseTransform, glm::vec4(r.direction, 0.0f));



	//for "Fast/Branchless Ray/Bounding Box Intersections" see(specifically part 2 dealing with NaN that arrise when ray is perfectly alligned with a slab)
	//https://tavianator.com/fast-branchless-raybounding-box-intersections/	
	//https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
	//calc inverse of ray slope's so we don't have to use divides during aabb testing
	const glm::vec3 dir_inv(1.f / rloc.direction);

	//needed for finding interp tri attributes later
	uint32_t closestTriIdx; 
	float triAreas[4];
	//Vertex vertices[3];

	//max stack size should be the maxDepth+1(node depth starts at 0) of the bvh we are traversing, largest I've seen is 32 (lucy.ply 1 tri/node)
	const uint32_t MSB_32BIT = 0x80000000;
	const uint32_t NUM_TRIS_MASK = ~MSB_32BIT;
	const uint32_t MAX_STACK_SIZE = 32;
	uint32_t stack[MAX_STACK_SIZE];
	int32_t stackPopIdx = -1;
	//push the first node index on the stack and update the stackPopIdx
	stack[++stackPopIdx] = model.modelInfo.startIdxBVHNode;
	float t = FLT_MAX;
	bool fullTraversal = false;//false

	int debugCount = -1;
	bool debugCase = false;
	//YOU CAN PRINTF
	while (stackPopIdx >= 0) {
		++debugCount;
		//pop off the stack to get the bvh index and then fetch the bvh data associated with this item
		const uint32_t bvhNodeIdx = stack[stackPopIdx--];
		const BVHNode nodeData = dev_BVHNodes[bvhNodeIdx];


		//check if leaf or inner node (msb of first payload member)
		const bool isInner = (MSB_32BIT & nodeData.payload.leaf.numTriangles) == 0;


		if (isInner) {//if inner perform isect testing of the two child aabb's. if both isect, push furthest on the stack, then closest (next pop will be the closer of the nodes)
			//one function call, pass both children AABB's, return t values, depending on hit combination and who's closer push onto the stack appropriately
			const float tLeft = isectAABB(nodeData.leftAABB, rloc, dir_inv);
			const float tRight = isectAABB(nodeData.rightAABB, rloc, dir_inv);

			if (tLeft != -1.f && tRight != -1.f) {//both hit by ray
				fullTraversal =  true; //hit both, need full traversal to not slip past something
				if (tLeft < tRight) {//left is closest, push last
					stack[++stackPopIdx] = nodeData.payload.inner.rightIdx;
					stack[++stackPopIdx] = nodeData.payload.inner.leftIdx;
				} else {//right is closest push last
					stack[++stackPopIdx] = nodeData.payload.inner.leftIdx;
					stack[++stackPopIdx] = nodeData.payload.inner.rightIdx;
				}
			} else if (tLeft != -1.f) { //only the left hit
				stack[++stackPopIdx] = nodeData.payload.inner.leftIdx;
			} else if (tRight != -1.f) {//only the right hit
				stack[++stackPopIdx] = nodeData.payload.inner.rightIdx;
			}

			//DEBUG INFO:REMOVE WHEN DONE
			//AABB aabbtris = nodeData.rightAABB; aabbtris.GrowAABB(nodeData.leftAABB);//diag should be all pos vals
			//const glm::vec3 d = aabbtris.max - aabbtris.min;
			//int axis = -1;
			//if (d.x > d.y && d.x > d.z) { axis = 0; } else if (d.y > d.z) { axis = 1; } else { axis = 2; }
			//const float overlap = nodeData.leftAABB.max[axis] - nodeData.rightAABB.min[axis];
			//const glm::vec3 leftHit = getPointOnRay(rloc, tLeft);
			//const glm::vec3 rightHit = getPointOnRay(rloc, tRight);

			//stop on both hit 
			//if (debugCount == 0 && tLeft != -1 && tRight != -1) {
			//	if (tLeft < tRight) {
			//		t = tLeft; debugCase = true;  break;
			//	} else {
			//		t = tRight; debugCase = true; break;
			//	}
			//}
			
			////exclusively left box (both hit continues)
			//if (debugCount == 0) {
			//	if (tLeft != -1 && tRight == -1) {
			//		t = tLeft; debugCase = true;  break;
			//	} else if (tLeft == -1 && tRight != -1) {
			//		t = FLT_MAX;  break;
			//	}
			//}

			////exclusively right box (both hit continues)
			//if (debugCount == 0) {
			//	if (tRight != -1 && tLeft == -1) {
			//		t = tRight; debugCase = true;  break;
			//	} else if (tRight == -1 && tLeft != -1) {
			//		t = FLT_MAX;  break;
			//	}
			//}
			//DEBUG INFO:REMOVE WHEN DONE


		} else {//else begin checking for triangle hits, take the closest amongst the hits, also get normal and uv information
			uint32_t numTris = NUM_TRIS_MASK & nodeData.payload.leaf.numTriangles;
			for (uint32_t i = 0; i < numTris; ++i) {
				const uint32_t triIdx = nodeData.payload.leaf.startIdx + i;
				const glm::ivec3 indices = dev_TriIndices[triIdx];
				float tmpTriAreas[4];
				float tTri = isectTriangle(indices, dev_TriVertices, rloc, tmpTriAreas);
				if (tTri != -1.f && tTri < t) {
					t = tTri;
					closestTriIdx = triIdx;
					//save off data for later use during attribute interpolation
					triAreas[0] = 1.f/tmpTriAreas[0];
					triAreas[1] = tmpTriAreas[1];
					triAreas[2] = tmpTriAreas[2];
					triAreas[3] = tmpTriAreas[3];
					//NOTE: will this buried global fetch hurt perf?
					//vertices[0] = dev_TriVertices[indices[0]];
					//vertices[1] = dev_TriVertices[indices[1]];
					//vertices[2] = dev_TriVertices[indices[2]];
				}
			}
			if (t != FLT_MAX && !fullTraversal) { break; }//if hit a triangle break out
		}// else (leaf node)
	}//while stackPopIdx is valid

	
	if (t == FLT_MAX) {//if t is still FLT_MAX return -1.f 
		return -1.f;
	} else { //otherwise find the intersectionPoint and normal with the valid t value (isectpoint - r.origin)
		//printf("\n%f closestTriHit", t);
		const glm::vec3 ploc = getPointOnRay(rloc, t);
		intersectionPoint = glm::vec3(model.transform * glm::vec4(ploc, 1));//WHY NOT JUST DO r.orgin + t*r.direction

		//if (debugCase) {
		//	normal = glm::vec3(0, 0, 0);
		//} else {
			const glm::ivec3 indices = dev_TriIndices[closestTriIdx];
			interpTriangleValues(indices, dev_TriVertices, triAreas, normal);
			normal = glm::normalize(model.invTranspose * normal);
		//}

		return t;
	}
}



__host__ __device__ float shapeIntersectionTest(const Geom& geom, const Ray& ray,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside,
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices) 
{
	if (geom.type == GeomType::CUBE) {
		return boxIntersectionTest(geom, ray, intersectionPoint, normal, outside);

	} else if (geom.type == GeomType::SPHERE) {
		return sphereIntersectionTest(geom, ray, intersectionPoint, normal,  outside);

	} else if (geom.type == GeomType::PLANE) {
		return planeIntersectionTest(geom, ray, intersectionPoint, normal, outside);
	} else if (geom.type == GeomType::MODEL) {
		return modelIntersectionTest(geom, ray, intersectionPoint, normal, outside, 
			dev_BVHNodes, dev_TriIndices, dev_TriVertices);
	} else {//doesn't recognize the type
		return -1.f;
	}
}


////////////////////////////////////////////
//////// FIND CLOSEST INTERSECTION /////////
////////////////////////////////////////////
__host__ __device__ glm::vec3 findClosestIntersection(const Ray& ray,
	const Geom* const geoms, int geoms_size, int& hit_geom_index,
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices) 
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
		} else if (geom.type == GeomType::MODEL) {
			t = modelIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside,
				dev_BVHNodes, dev_TriIndices, dev_TriVertices);
		} else {//doesn't recognize the type
			t = -1.f;
		}

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (0.f < t && t < t_min) {
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
