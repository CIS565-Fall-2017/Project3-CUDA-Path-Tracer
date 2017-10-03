#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "sceneStructs.h"
#include "utilities.h"


__device__ glm::mat4 dev_buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
	float pi = 3.14159f;
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)pi / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)pi / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)pi / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
}

__global__ void kernCheckBounds(int n, Geom* geoms) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > n) {
		return;
	}
	Geom &geom = geoms[idx];
	//printf("id: %i, new idx: %i, old idx %i xmin: %f xmax: %f\n", geom.type, idx, geom.id, geom.minb.x,geom.maxb.x);
}
__global__ void kernCalculateBounds(int n, Geom* geoms) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > n) {
		return;
	}
	Geom &geom = geoms[idx];
	if (geom.type == TRIANGLE) {
		glm::vec3 maxb;
		glm::vec3 minb;
		glm::vec3 p1 = multiplyMV(geom.transform, glm::vec4(geom.points[0], 1.0f));
		glm::vec3 p2 = multiplyMV(geom.transform, glm::vec4(geom.points[1], 1.0f));
		glm::vec3 p3 = multiplyMV(geom.transform, glm::vec4(geom.points[2], 1.0f));
		maxb.x = glm::max(glm::max(p1.x, p2.x), p3.x);
		maxb.y = glm::max(glm::max(p1.y, p2.y), p3.y);
		maxb.z = glm::max(glm::max(p1.z, p2.z), p3.z);
		minb.x = glm::min(glm::min(p1.x, p2.x), p3.x);
		minb.y = glm::min(glm::min(p1.y, p2.y), p3.y);
		minb.z = glm::min(glm::min(p1.z, p2.z), p3.z);
		geom.maxb = maxb;
		geom.minb = minb;
		geom.midpoint = (maxb + minb) / 2.0f;

		//surface area calc
		float a = glm::distance(p1, p2);
		float b = glm::distance(p3, p2);
		float c = glm::distance(p1, p3);
		float s = (a + b + c) / 2.0f;
		geom.surface_area = sqrt(s* (s - a) * (s - b) * (s - c));
	}
	else if (geom.type == SPHERE) {
		glm::vec3 maxb;
		glm::vec3 minb;
		glm::vec3 p1 = multiplyMV(geom.transform, glm::vec4(0.5f, 0.0f, 0.0f, 1.0f));
		glm::vec3 p2 = multiplyMV(geom.transform, glm::vec4(0.0f, 0.5f, 0.0f, 1.0f));
		glm::vec3 p3 = multiplyMV(geom.transform, glm::vec4(0.0f, 0.0f, 0.5f, 1.0f));
		glm::vec3 p4 = multiplyMV(geom.transform, glm::vec4(-0.5f, 0.0f, 0.0f, 1.0f));
		glm::vec3 p5 = multiplyMV(geom.transform, glm::vec4(0.0f, -0.5f, 0.0f, 1.0f));
		glm::vec3 p6 = multiplyMV(geom.transform, glm::vec4(0.0f, 0.0f, -0.5f, 1.0f));
		for (int i = 0; i < 3; i++) {
			maxb[i] = glm::max(glm::max(glm::max(glm::max(glm::max(p1[i], p2[i]), p3[i]), p4[i]), p5[i]), p6[i]);
			minb[i] = glm::min(glm::min(glm::min(glm::min(glm::min(p1[i], p2[i]), p3[i]), p4[i]), p5[i]), p6[i]);
		}
		geom.maxb = maxb;
		geom.minb = minb;
		geom.midpoint = (maxb + minb) / 2.0f;
		glm::vec3 scale = geom.scale;
		float a = scale.x;
		float b = scale.y;
		float c = scale.z;
		geom.surface_area = 4 * PI*powf(((powf(a*b, 1.6) + powf(a*c, 1.6) + powf(c*b, 1.6)) / 3.0f), 1.0f / 1.6f);
	}
	else if (geom.type == CUBE) {
		glm::vec3 maxb;
		glm::vec3 minb;
		glm::vec3 p1 = multiplyMV(geom.transform, glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
		glm::vec3 p2 = multiplyMV(geom.transform, glm::vec4(0.5f, -0.5f, 0.5f, 1.0f));
		glm::vec3 p3 = multiplyMV(geom.transform, glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f));
		glm::vec3 p4 = multiplyMV(geom.transform, glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f));
		glm::vec3 p5 = multiplyMV(geom.transform, glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f));
		glm::vec3 p6 = multiplyMV(geom.transform, glm::vec4(0.5f, 0.5f, -0.5f, 1.0f));
		glm::vec3 p7 = multiplyMV(geom.transform, glm::vec4(0.5f, -0.5f, -0.5f, 1.0f));
		glm::vec3 p8 = multiplyMV(geom.transform, glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f));
		for (int i = 0; i < 3; i++) {
			maxb[i] = glm::max(glm::max(glm::max(glm::max(glm::max(glm::max(glm::max(p1[i], p2[i]), p3[i]), p4[i]), p5[i]), p6[i]), p7[i]), p8[i]);
			minb[i] = glm::min(glm::min(glm::min(glm::min(glm::min(glm::min(glm::min(p1[i], p2[i]), p3[i]), p4[i]), p5[i]), p6[i]), p7[i]), p8[i]);
		}
		geom.maxb = maxb;
		geom.minb = minb;
		geom.midpoint = (maxb + minb) / 2.0f;
		glm::vec3 scale = geom.scale;
		geom.surface_area = scale.x * scale.y * scale.z;
	}
	//printf("idx: %i, max %f %f %f\n", idx, geom.maxb.x, geom.maxb.y, geom.maxb.z);
	//printf("idx: %i, min %f %f %f\n", idx, geom.minb.x, geom.minb.y, geom.minb.z);
	//printf("idx: %i, mid %f %f %f, type %i, sa: %f\n", idx, geom.midpoint.x, geom.midpoint.y, geom.midpoint.z, geom.type, geom.surface_area);
	
	//printf("id: %i, idx: %i\n", geom.type, idx);
}

struct testpred {
	__host__ __device__ bool operator() (const PathSegment& pathSegment) {
		return (pathSegment.remainingBounces > 0);
	}
};
struct less_than_axis {
	less_than_axis(int x, float val) : axis(x), val(val) {}
	__host__ __device__ bool operator()(const Geom & geom) const { return geom.midpoint[axis] < val; }

private:
	int axis;
	float val;
};

struct get_maxb {
	__host__ __device__
		glm::vec3 operator()(const Geom& g) {
		return g.maxb;
	}
};

struct get_max_vec {
	__host__ __device__
		glm::vec3 operator()(const glm::vec3 g1,
			const glm::vec3 g2) {
		glm::vec3 ret;
		g1.x > g2.x ? ret.x = g1.x : ret.x = g2.x;
		g1.y > g2.y ? ret.y = g1.y : ret.y = g2.y;
		g1.z > g2.z ? ret.z = g1.z : ret.z = g2.z;
		return ret;
	}
};

struct get_min_vec {
	__host__ __device__
		glm::vec3 operator()(const glm::vec3 g1,
			const glm::vec3 g2) {
		glm::vec3 ret;
		g1.x < g2.x ? ret.x = g1.x : ret.x = g2.x;
		g1.y < g2.y ? ret.y = g1.y : ret.y = g2.y;
		g1.z < g2.z ? ret.z = g1.z : ret.z = g2.z;
		return ret;
	}
};

struct get_minb {
	__host__ __device__
		glm::vec3 operator()(const Geom& g) {
		return g.minb;
	}
};

struct get_sa {
	__host__ __device__
		float operator()(const Geom& g) {
		return g.surface_area;
	}
};

struct sum_sa {
	__host__ __device__
		float operator()(const float g1,
			const float g2) {
		return g1 + g2;
	}
};

__global__ void kernInitBVH(int n, int *BVHIndices) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > n) {
		return;
	}
	BVHIndices[idx] = 0;
}

__global__ void kernFillBVH(int idx, int *BVHIndices, BVHNode * BVHnodes) {

}

__global__ void kernSetBVHTransform(int n, BVHNode * BVHnodes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%i \n", idx);
	if (idx > n) {
		return;
	}
	BVHNode & node = BVHnodes[idx];
	//printf("node: %i %f %f\n", idx, node.minb.x ,node.maxb.x);
	glm::vec3 translate = (node.maxb + node.minb) / 2.0f;
	glm::vec3 scale = (node.maxb - node.minb);
	node.transform = dev_buildTransformationMatrix(
		translate, glm::vec3(0.0f), scale);
	node.inverseTransform = glm::inverse(node.transform);
	node.invTranspose = glm::inverseTranspose(node.transform);
	/*printf("idx: %i, trans: %f %f %f, scale %f %f %f\n", idx, translate.x, translate.y, translate.z,
		scale.x, scale.y, scale.z);*/
}

__host__ __device__ float BVHIntersectionTest(BVHNode box, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
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
		intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
		return glm::length(r.origin - intersectionPoint);
	}
	return -1;
}

__global__ void computeBVHIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, ShadeableIntersection * intersections
	, BVHNode * BVHNodes
	, int num_bvh
	, Geom* geoms
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		glm::vec2 geom_ranges[300];
		int geom_idx = 0;
		PathSegment pathSegment = pathSegments[path_index];
		if (pathSegment.pixelIndex == 400) {
			int pixelIndex = pathSegment.pixelIndex;
			pixelIndex++;
		}
		BVHNode* stack[300];
		BVHNode* * stackPtr = stack;
		*stackPtr++ = NULL; // push

		bool loutside = true;
		bool routside = true;

		glm::vec3 ltmp_intersect;
		glm::vec3 ltmp_normal;
		glm::vec3 rtmp_intersect;
		glm::vec3 rtmp_normal;

		// naive parse through global geoms

		BVHNode * curr_bvh = &BVHNodes[0];
		bool loop = true;

		while (curr_bvh != NULL) {
			BVHNode lchild = BVHNodes[curr_bvh->child1id];
			BVHNode rchild = BVHNodes[curr_bvh->child2id];
			float l_intersect = BVHIntersectionTest(lchild, pathSegment.ray, ltmp_intersect, ltmp_normal, loutside);
			float r_intersect = BVHIntersectionTest(rchild, pathSegment.ray, rtmp_intersect, rtmp_normal, routside);
			
			if (l_intersect > 0.0f && lchild.is_leaf) {
					geom_ranges[geom_idx++] = glm::vec2(lchild.start, lchild.end);
				//queue up geoms for intersection test
			}
			if (r_intersect > 0.0f && rchild.is_leaf) {
				geom_ranges[geom_idx++] = glm::vec2(rchild.start, rchild.end);
				//queue up geoms for intersection test
			}

			bool traverseL = (l_intersect > 0.0f && !lchild.is_leaf);
			bool traverseR = (r_intersect > 0.0f && !rchild.is_leaf);

			if (!traverseL && !traverseR) {
				curr_bvh = *--stackPtr; // pop
			}
			else
			{
				int r_id = curr_bvh->child2id;
				curr_bvh = ((l_intersect > 0.0f && !lchild.is_leaf)) ? &BVHNodes[curr_bvh->child1id] : &BVHNodes[curr_bvh->child2id];
				if ((l_intersect > 0.0f && !lchild.is_leaf) && (r_intersect > 0.0f && !rchild.is_leaf)) {
					*stackPtr++ = &BVHNodes[r_id]; // push
				}
			}
		}

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		for (int i = 0; i < geom_idx; i++) {
			int start = geom_ranges[i][0];
			int end = geom_ranges[i][1];
			for (int g = start; g <= end; g++) {
				Geom & geom = geoms[g];

				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				}
				else if (geom.type == TRIANGLE) {
					t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				}
				// TODO: add more intersection tests here... triangle? metaball? CSG?

				// Compute the minimum t from the intersection tests to determine what
				// scene geometry object was hit first.
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = g;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}
			}

			if (hit_geom_index == -1)
			{
				intersections[path_index].t = -1.0f;
			}
			else
			{
				//The ray hits something
				intersections[path_index].t = t_min;
				intersections[path_index].materialId = geoms[hit_geom_index].materialid;
				intersections[path_index].surfaceNormal = normal;
			}
		}
	}
}

__global__ void computeBVHDebugIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, ShadeableIntersection * intersections
	, BVHNode * BVHNodes
	, int num_bvh
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		BVHNode* stack[64];
		BVHNode* * stackPtr = stack;
		*stackPtr++ = NULL; // push

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool loutside = true;
		bool routside = true;

		glm::vec3 ltmp_intersect;
		glm::vec3 ltmp_normal;
		glm::vec3 rtmp_intersect;
		glm::vec3 rtmp_normal;

		// naive parse through global geoms

		BVHNode * curr_bvh = &BVHNodes[0];
		bool loop = true;

		while (curr_bvh != NULL) {
			BVHNode lchild = BVHNodes[curr_bvh->child1id];
			BVHNode rchild = BVHNodes[curr_bvh->child2id];
			float l_intersect = BVHIntersectionTest(lchild, pathSegment.ray, ltmp_intersect, ltmp_normal, loutside);
			float r_intersect = BVHIntersectionTest(rchild, pathSegment.ray, rtmp_intersect, rtmp_normal, routside);

			if (l_intersect > 0.0f && lchild.is_leaf) {
				intersections[path_index].t = l_intersect;
				intersections[path_index].surfaceNormal = glm::vec3((float)curr_bvh->id / (float)num_bvh, 0.0f, 0.0f);
				//queue up geoms for intersection test
			}
			if (r_intersect > 0.0f && rchild.is_leaf) {
				if (r_intersect > l_intersect) {
					intersections[path_index].t = l_intersect;
					intersections[path_index].surfaceNormal = glm::vec3((float)curr_bvh->id / (float)num_bvh, 0.0f, 0.0f);
				}//queue up geoms for intersection test
			}

			bool traverseL = (l_intersect > 0.0f && !lchild.is_leaf);
			bool traverseR = (r_intersect > 0.0f && !rchild.is_leaf);

			if (!traverseL && !traverseR) {
				curr_bvh = *--stackPtr; // pop
			}
			else
			{
				int r_id = curr_bvh->child2id;
				curr_bvh = ((l_intersect > 0.0f && !lchild.is_leaf)) ? &BVHNodes[curr_bvh->child1id] : &BVHNodes[curr_bvh->child1id];
				if ((l_intersect > 0.0f && !lchild.is_leaf) && (r_intersect > 0.0f && !rchild.is_leaf)) {
					*stackPtr++ = &BVHNodes[r_id]; // push
				}
			}
		}

	}
}