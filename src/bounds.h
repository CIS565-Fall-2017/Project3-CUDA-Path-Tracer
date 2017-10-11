#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <vector>

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

class Bounds3f
{
public:
    Bounds3f()
        : min(glm::vec3(0.f)), max(glm::vec3(0.f))
    {}
    Bounds3f(const glm::vec3& min, const glm::vec3& max)
        : min(min), max(max)
    {}
    Bounds3f(const glm::vec3& p)
        : min(p), max(p)
    {}

    // Returns a vector representing the diagonal of the box
    glm::vec3 Diagonal() const { return max - min; }

    // Returns the index of the axis with the largest length
    int MaximumExtent() const
    {
        glm::vec3 d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    // Returns the position of a point *relative*
    // to the min and max corners of the box.
    // This ranges from (0, 0, 0) to (1, 1, 1)
    // where these two extremes represent the
    // values the result would have at the min and
    // max corners of the box
    glm::vec3 Offset(const glm::vec3 &p) const
    {
        glm::vec3 o = p - min;
        if (max.x > min.x) o.x /= max.x - min.x;
        if (max.y > min.y) o.y /= max.y - min.y;
        if (max.z > min.z) o.z /= max.z - min.z;
        return o;
    }

    // Transforms this Bounds3f by the input Transform and
    // also returns a Bounds3f representing our new boundaries.
    // Transforming a Bounds3f is equivalent to creating a Bounds3f
    // that encompasses the non-axis-aligned box resulting from
    // transforming this Bounds3f's eight corners
//    Bounds3f Apply(const glm::mat4& tr);

    // Returns the surface area of this bounding box
    float SurfaceArea() const;

	__host__ __device__ inline bool Inside(const glm::vec3 &p) const {
		return (p.x >= min.x && p.x <= max.x &&
			p.y >= min.y && p.y <= max.y &&
			p.z >= min.z && p.z <= max.z);
	}

	__host__ __device__ bool Intersect(const Ray &r, float* t) const
	{
		Ray r_loc = r;

		float t_n = FLT_MIN;
		float t_f = FLT_MAX;
		for (int i = 0; i < 3; i++) {

			float minValue, maxValue;

			if (i == 0) {
				minValue = min.x;
				maxValue = max.x;
			}
			if (i == 1) {
				minValue = min.y;
				maxValue = max.y;
			}
			if (i == 2) {
				minValue = min.z;
				maxValue = max.z;
			}


			//Ray parallel to slab check
			if (r_loc.direction[i] == 0) {
				if (r_loc.origin[i] < minValue || r_loc.origin[i] > maxValue) {
					return false;
				}
			}
			//If not parallel, do slab intersect check
			float t0 = (minValue - r_loc.origin[i]) / r_loc.direction[i];
			float t1 = (maxValue - r_loc.origin[i]) / r_loc.direction[i];
			if (t0 > t1) {
				float temp = t1;
				t1 = t0;
				t0 = temp;
			}
			if (t0 > t_n) {
				t_n = t0;
			}
			if (t1 < t_f) {
				t_f = t1;
			}
		}
		if (t_n < t_f)
		{
			float t_result = t_n > 0 ? t_n : t_f;
			if (t_result < 0) {

				// negative t values are valid if and only if the ray's origin lies within the bounding box
				if (Inside(r_loc.origin)) {
					(*t) = t_result;
					return true;
				}

				return false;
			}

			(*t) = t_result;
			return true;
		}
		else {
			//If t_near was greater than t_far, we did not hit the cube
			return false;
		}
	}

    //Used in IntersectP
	__host__ __device__ inline const glm::vec3& operator[](int i) const {
        return (i == 0) ? min : max;
    }

    //Used in IntersectP
	__host__ __device__ inline glm::vec3 operator[](int i) {
        return (i == 0) ? min : max;
    }

    //Used in IntersectP
	__host__ __device__ static inline float gamma(int n) {
        return (n *std::numeric_limits<float>::epsilon() * 0.5) / (1 - n * std::numeric_limits<float>::epsilon() * 0.5);
    }

    //Used in BVH intersect
	__host__ __device__
    inline bool IntersectP(const Ray &ray, const glm::vec3 &invDir,
                           const int dirIsNeg[3]) const {
        const Bounds3f &bounds = *this;
        // Check for ray intersection against $x$ and $y$ slabs
        float tMin = (bounds[dirIsNeg[0]].x - ray.origin.x) * invDir.x;
        float tMax = (bounds[1 - dirIsNeg[0]].x - ray.origin.x) * invDir.x;
        float tyMin = (bounds[dirIsNeg[1]].y - ray.origin.y) * invDir.y;
        float tyMax = (bounds[1 - dirIsNeg[1]].y - ray.origin.y) * invDir.y;

        // Update _tMax_ and _tyMax_ to ensure robust bounds intersection
        tMax *= 1 + 2 * gamma(3);
        tyMax *= 1 + 2 * gamma(3);
        if (tMin > tyMax || tyMin > tMax) return false;
        if (tyMin > tMin) tMin = tyMin;
        if (tyMax < tMax) tMax = tyMax;

        // Check for ray intersection against $z$ slab
        float tzMin = (bounds[dirIsNeg[2]].z - ray.origin.z) * invDir.z;
        float tzMax = (bounds[1 - dirIsNeg[2]].z - ray.origin.z) * invDir.z;

        // Update _tzMax_ to ensure robust bounds intersection
        tzMax *= 1 + 2 * gamma(3);
        if (tMin > tzMax || tzMin > tMax) return false;
        if (tzMin > tMin) tMin = tzMin;
        if (tzMax < tMax) tMax = tzMax;
        return (tMax > 0);
    }


    // Check whether two bounding box intersects or nor
    inline bool IntersectBoundingBox(const glm::vec3& b_min, const glm::vec3& b_max){

        float new_min_X = glm::max(min.x, b_min.x);
        float new_min_Y = glm::max(min.y, b_min.y);
        float new_min_Z = glm::max(min.z, b_min.z);

        float new_max_X = glm::min(max.x, b_max.x);
        float new_max_Y = glm::min(max.y, b_max.y);
        float new_max_Z = glm::min(max.z, b_max.z);

        return (new_min_X <= new_max_X &&
                new_min_Y <= new_max_Y &&
                new_min_Z <= new_max_Z);
    }



    // Used for k-d tree intersection
    //inline bool IntersectAndGetTwoTValue(const Ray &ray, float *hitt0, float *hitt1) const {

    //    float t0 = 0.f;
    //    float t1 = 1000000.f;

    //    for(int i = 0; i < 3; i++){
    //        // Update interval for ith bounding box slab
    //        float invRayDir = 1.0f / ray.direction[i];
    //        float tNear = (min[i] - ray.origin[i]) * invRayDir;
    //        float tFar  = (max[i] - ray.origin[i]) * invRayDir;

    //        // Update parametric interval from slab intersection t values
    //        if(tNear > tFar) std::swap(tNear, tFar);

    //        t0 = tNear > t0 ? tNear : t0;
    //        t1 = tFar  < t1 ? tFar  : t1;
    //        if (t0 > t1) return false;
    //    }

    //    if (hitt0) *hitt0 = t0;
    //    if (hitt1) *hitt1 = t1;

    //    return true;
    //}


    glm::vec3 min, max;
};

Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2);
Bounds3f Union(const Bounds3f& b1, const glm::vec3& p);
Bounds3f Union(const Bounds3f& b1, const glm::vec4& p);