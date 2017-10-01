#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

typedef float Float;
typedef glm::vec3 Color3f;
typedef glm::vec3 Point3f;
typedef glm::vec3 Normal3f;
typedef glm::vec2 Point2f;
typedef glm::ivec2 Point2i;
typedef glm::ivec3 Point3i;
typedef glm::vec3 Vector3f;
typedef glm::vec2 Vector2f;
typedef glm::ivec2 Vector2i;
typedef glm::mat4 Matrix4x4;
typedef glm::mat3 Matrix3x3;

// Global constants. You may not end up using all of these.
#define ShadowEpsilon 0.0001f
#define RayEpsilon 0.000005f
#define RayMarchingEpsilon 0.1f
#define Pi 3.14159265358979323846f
#define TwoPi 6.28318530717958647692f
#define InvPi 0.31830988618379067154f
#define Inv2Pi 0.15915494309189533577f
#define Inv4Pi 0.07957747154594766788f
#define PiOver2 1.57079632679489661923f
#define PiOver4 0.78539816339744830961f
#define Sqrt2 1.41421356237309504880f
#define OneMinusEpsilon 0.99999994f
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
    //return r.origin + (t - .0001f) * glm::normalize(r.direction);

	return r.origin + t * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__ glm::vec3 getTextColor(int width, int height, int beginIndex, glm::vec2 UV, glm::vec3* imageData)
{
	float fw = (width)* (UV.x - glm::floor(UV.x));
	float fh = (height) *  (UV.y -  glm::floor(UV.y));

	//float fw = (width)* (1.0f - (UV.y - glm::floor(UV.y)));
	//float fh = (height) *  (UV.x - glm::floor(UV.x));

	int firstW = (int)fw;
	int secondW = (firstW + 1) < (width - 1) ? (firstW + 1) : (width - 1);

	int firstH = (int)fh;
	int secondH = (firstH + 1) < (height - 1) ? (firstH + 1) : (height - 1);

	float x_gap = fw - firstW;
	float y_gap = fh - firstH;

	//Bi-linear

	glm::vec3 color01 = glm::mix(imageData[firstW + firstH*width + beginIndex], imageData[secondW + firstH*width + beginIndex] , x_gap);
	glm::vec3 color02 = glm::mix(imageData[firstW + secondH*width + beginIndex], imageData[secondW + secondH*width + beginIndex], x_gap);

	return glm::mix(color01, color02, y_gap);
}

__host__ __device__ glm::mat3 Cube_ComputeTBN(glm::vec3 nor)
{
	
	glm::vec3 localNormal = nor;

	glm::vec3 tan;
	glm::vec3 bit;


	float absX = glm::abs(localNormal.x);
	float absY = glm::abs(localNormal.y);
	float absZ = glm::abs(localNormal.z);


	if (absX > absY && absX > absZ)
	{
		if (absX > 0.0f)
		{
			tan = glm::vec3(0.0f, 0.0f, -1.0f);
			bit = glm::vec3(0.0f, 1.0f, 0.0f);
		}
		else
		{
			tan = glm::vec3(0.0f, 0.0f, 1.0f);
			bit = glm::vec3(0.0f, 1.0f, 0.0f);
		}
	}
	else if (absY > absX && absY > absZ)
	{
		if (absY > 0.0f)
		{
			tan = glm::vec3(1.0f, 0.0f, 0.0f);
			bit = glm::vec3(0.0f, 0.0f, -1.0f);
		}
		else
		{
			tan = glm::vec3(1.0f, 0.0f, 0.0f);
			bit = glm::vec3(0.0f, 0.0f, 1.0f);
		}
	}
	else
	{
		if (absZ > 0.0f)
		{
			tan = glm::vec3(1.0f, 0.0f, 0.0f);
			bit = glm::vec3(0.0f, 1.0f, 0.0f);
		}
		else
		{
			tan = glm::vec3(0.0f, 0.0f, 1.0f);
			bit = glm::vec3(0.0f, 1.0f, 0.0f);
		}
	}


	glm::mat3 tbn;

	tbn[0] = tan;
	tbn[1] = bit;
	tbn[2] = nor;

	return tbn;

}

__host__ __device__  glm::vec2 Cube_GetUVCoordinates(const glm::vec3 &point)
{
	glm::vec3 abs;
	
	abs.x = glm::abs(point.x) < 0.5f ? glm::abs(point.x) : 0.5f;
	abs.y = glm::abs(point.y) < 0.5f ? glm::abs(point.y) : 0.5f;
	abs.z = glm::abs(point.z) < 0.5f ? glm::abs(point.z) : 0.5f;

	glm::vec2 UV;//Always offset lower-left corner
	if (abs.x > abs.y && abs.x > abs.z)
	{
		UV = glm::vec2(point.z + 0.5f, point.y + 0.5f);		
	}
	else if (abs.y > abs.x && abs.y > abs.z)
	{
		UV = glm::vec2(point.x + 0.5f, 1.0f - (point.z + 0.5f));		
	}
	else
	{
		UV = glm::vec2(point.x + 0.5f, point.y + 0.5f);		
	}
	return UV;
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
__host__ __device__ float boxIntersectionTest(Geom box, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2 &uv, bool &outside,
	int normalTexID, Image* ImageHeader, glm::vec3* imageData)
{
	Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
	
    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
	{
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
		{
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? 1.0f : -1.0f;
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

    if (tmax >= tmin && tmax > 0)
	{
        outside = true;

        if (tmin <= 0)
		{
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }

		glm::vec3 intersectionPointLocal = getPointOnRay(q, tmin);
        intersectionPoint = multiplyMV(box.transform, glm::vec4(intersectionPointLocal, 1.0f));		
		uv = Cube_GetUVCoordinates(intersectionPointLocal);

		if (normalTexID >= 0)
		{
			glm::mat3 tbn = Cube_ComputeTBN(tmin_n);

			const Image &header = ImageHeader[normalTexID];

			glm::vec3 normalColor = getTextColor(header.width, header.height, header.beginIndex, uv, imageData);
			normalColor = glm::normalize(normalColor*2.0f - glm::vec3(1.0f));

			normal = glm::normalize(tbn * normalColor);
			//normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(normal, 0.0f)));
		}
		else		
			normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));


		
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

__host__ __device__ glm::mat3 Sphere_ComputeTBN(glm::vec3 nor, glm::vec3 worldP, glm::mat3 rotMat)
{
	glm::vec3 tan;
	glm::vec3 bit;

	
	tan = glm::normalize(rotMat * glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), (glm::normalize(worldP))));
	bit = glm::normalize(glm::cross(nor, tan));

	glm::mat3 tbn;

	tbn[0] = tan;
	tbn[1] = bit;
	tbn[2] = nor;

	return tbn;
}

__host__ __device__ glm::vec2 Sphere_GetUVCoordinates(const glm::vec3 &point)
{
	glm::vec3 p = glm::normalize(point);
	float phi = glm::atan(p.z , p.x);
	if (phi < 0.0f)
	{
		phi += 6.283185f;
	}
	float theta = glm::acos(p.y);
	return glm::vec2(1.0f - phi / 6.283185f, 1.0f - theta / 3.141593f);
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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2 &uv, bool &outside,
	int normalTexID, Image* ImageHeader, glm::vec3* imageData) {
    float radius = .5f;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - glm::pow(radius, 2.0f));
    
	
	if (radicand < 0) {
        return -1.0f;
    }

	
    float squareRoot = glm::sqrt(radicand);
	
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1.0f;
    } else if (t1 > 0 && t2 > 0) {
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
	uv = Sphere_GetUVCoordinates(objspaceIntersection);

	if (normalTexID >= 0)
	{
		

		glm::mat3 tbn = Sphere_ComputeTBN( glm::normalize(objspaceIntersection), intersectionPoint, (glm::mat3)sphere.rotationMat);

		const Image &header = ImageHeader[normalTexID];

		glm::vec3 normalColor = getTextColor(header.width, header.height, header.beginIndex, uv, imageData);
		normalColor = glm::normalize(normalColor*2.0f - glm::vec3(1.0f));

		normal = glm::normalize(tbn * normalColor);
		//normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(normal, 0.f)));
	
	}
	else
		normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

   
	
    if (!outside) {
        normal = -normal;		
    }
	

	return  glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool BBIntersect(const Ray &r, AABB boundingBox, float* t)
{
	//TODO
	//Transform the ray

	float t_n = -FLT_MAX;
	float t_f = FLT_MAX;
	{
		//Ray parallel to slab check
		if (r.direction[0] == 0)
		{
			if (r.origin[0] < boundingBox.min.x || r.origin[0] > boundingBox.max.x) {
				return false;
			}
		}
		//If not parallel, do slab intersect check
		float t0 = (boundingBox.min.x - r.origin[0]) / r.direction[0];
		float t1 = (boundingBox.max.x - r.origin[0]) / r.direction[0];
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


		//Ray parallel to slab check
		if (r.direction[1] == 0) {
			if (r.origin[1] < boundingBox.min.y || r.origin[1] > boundingBox.max.y) {
				return false;
			}
		}
		//If not parallel, do slab intersect check
		t0 = (boundingBox.min.y - r.origin[1]) / r.direction[1];
		t1 = (boundingBox.max.y - r.origin[1]) / r.direction[1];
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


		//Ray parallel to slab check
		if (r.direction[2] == 0) {
			if (r.origin[2] < boundingBox.min.z || r.origin[2] > boundingBox.max.z) {
				return false;
			}
		}
		//If not parallel, do slab intersect check
		t0 = (boundingBox.min.z - r.origin[2]) / r.direction[2];
		t1 = (boundingBox.max.z - r.origin[2]) / r.direction[2];
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
		if ((r.origin[0] >= boundingBox.min.x && r.origin[0] <= boundingBox.max.x) &&
			(r.origin[1] >= boundingBox.min.y && r.origin[1] <= boundingBox.max.y) &&
			(r.origin[2] >= boundingBox.min.z && r.origin[2] <= boundingBox.max.z))
		{
			*t = t_n;
		}
		else
		{
			float result_t = t_n > 0 ? t_n : t_f;
			if (result_t < 0)
				return false;


			*t = result_t;
		}


		return true;
	}
	else
	{
		//If t_near was greater than t_far, we did not hit the cube
		return false;
	}
}

__host__ __device__ glm::mat3 TriangleComputeTriangleTBN(const Triangle &triangle, glm::vec3 planeNormal)
{
	//*nor = GetNormal(P);
	//TODO: Compute tangent and bitangent based on UV coordinates.
	
	float e0x = triangle.position1.x - triangle.position0.x;
	float e1x = triangle.position2.x - triangle.position0.x;

	float e0y = triangle.position1.y - triangle.position0.y;
	float e1y = triangle.position2.y - triangle.position0.y;

	float e0z = triangle.position1.z - triangle.position0.z;
	float e1z = triangle.position2.z - triangle.position0.z;
	
	float u0 = triangle.texcoord1.x - triangle.texcoord0.x;
	float u1 = triangle.texcoord2.x - triangle.texcoord0.x;

	float v0 = triangle.texcoord1.y - triangle.texcoord0.y;
	float v1 = triangle.texcoord2.y - triangle.texcoord0.y;

	float dino = u0 * v1 - v0 * u1;

	glm::vec3 tan;
	glm::vec3 bit;
	glm::vec3 nor = planeNormal;

	if (dino != 0.0f)
	{
		float r = 1.0f / (dino);
		tan = glm::normalize(glm::vec3((v1 * e0x - v0 * e1x) * r, (v1 * e0y - v0 * e1y) * r, (v1 * e0z - v0 * e1z) * r));
		bit = glm::normalize(glm::cross(nor, tan));
		tan = glm::normalize(glm::cross(bit, nor));
	}
	else
	{

		tan = glm::vec3(1.0f, 0.0f, 0.0f);
		bit = glm::normalize(glm::cross(nor, tan));
		tan = glm::normalize(glm::cross(bit, nor));
	}

	// Calculate handedness
	glm::vec3 fFaceNormal;
	fFaceNormal = glm::cross(glm::normalize(glm::vec3(e0x, e0y, e0z)), glm::normalize(glm::vec3(e1x, e1y, e1z)));

	fFaceNormal = glm::normalize(fFaceNormal);

	//U flip
	if (glm::dot(glm::cross(tan, bit), fFaceNormal) < 0.0f)
	{
		tan = -(tan);
	}

	glm::mat3 tbn;

	tbn[0] = tan;
	tbn[1] = bit;
	tbn[2] = nor;

	return tbn;
}




__host__ __device__ float triangleIntersectionTest(Geom geom, Triangle triangle, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2 &uv, bool &outside,
	int normalTexID, Image* ImageHeader, glm::vec3* imageData)
{
	Ray q;
	q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

	//glm::vec3 planeNormal = triangle.planeNormal;

	if (glm::dot(triangle.planeNormal, q.direction) > 0.0f)
		return -1.0f;

	//1. Ray-plane intersection
	float t = glm::dot(triangle.planeNormal, (triangle.position0 - q.origin)) / glm::dot(triangle.planeNormal, q.direction);
	if (t < 0) return -1.0f;



	glm::vec3 P = q.origin + t * q.direction;
	//2. Barycentric test
	float S = 0.5f * glm::length(glm::cross(triangle.position0 - triangle.position1, triangle.position0 - triangle.position2));
	float s1 = 0.5f * glm::length(glm::cross(P - triangle.position1, P - triangle.position2));
	float s2 = 0.5f * glm::length(glm::cross(P - triangle.position2, P - triangle.position0));
	float s3 = 0.5f * glm::length(glm::cross(P - triangle.position0, P - triangle.position1));
	float sum = s1 + s2 + s3;

	//if (s1 >= 0 && s1 <= 1 && s2 >= 0 && s2 <= 1 && s3 >= 0 && s3 <= 1 && (sum >= S*0.99999f && sum <= S*1.00001f))
	if (sum <= S*1.00001f)
	{
		intersectionPoint = multiplyMV(geom.transform, glm::vec4(P, 1.f));

		glm::vec3 smoothedNormal = glm::normalize(triangle.normal0 * (s1 / sum) + triangle.normal1 * (s2 / sum) + triangle.normal2 * (s3 / sum));
		uv = triangle.texcoord0 * (s1 / sum) + triangle.texcoord1 * (s2 / sum) + triangle.texcoord2 * (s3 / sum);

		if (normalTexID >= 0)
		{
			

			glm::mat3 tbn = TriangleComputeTriangleTBN(triangle, smoothedNormal);

			const Image &header = ImageHeader[normalTexID];

			glm::vec3 normalColor = getTextColor(header.width, header.height, header.beginIndex, uv, imageData);

			normalColor = glm::normalize(normalColor*2.0f - glm::vec3(1.0f));			
			normal = glm::normalize(tbn * normalColor);
			//normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.0f)));
		}
		else
			normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(smoothedNormal, 0.0f)));
		

		return glm::length(r.origin - intersectionPoint);
	}
	
	return -1;
}


/*
__host__ __device__ float traverseOctree(const Ray &r, const Octree *octreees, int thisOctreeNum, Geom geom, const Triangle* triangles, glm::vec3 &intersectionPoint, glm::vec3 &normal, float t_min)
{
	float t = -1.0f;
	float t_min_saved = t_min;

	glm::vec3 intersectionPoint_saved = intersectionPoint;
	glm::vec3 normal_saved = normal;


	bool bOutside;

	const Octree &thisOctree = octreees[thisOctreeNum];

	if (BBIntersect(r, thisOctree.boundingBox, &t))
	{
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		if (t_min <= t)
			return -1.0f;

		

		int index = thisOctree.firstElement;
		while (index > -1)
		{
			t = triangleIntersectionTest(geom, triangles[index], r, tmp_intersect, tmp_normal, bOutside);

			if (t > 0.0f && t_min_saved > t)
			{
				t_min_saved = t;
				intersectionPoint_saved = tmp_intersect;
				normal_saved = tmp_normal;
			}

			index = triangles[index].nextTriangleID;
		}

		

		//child later
		if (thisOctree.child_01 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_01, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		

		if (thisOctree.child_02 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_02, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (thisOctree.child_03 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_03, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (thisOctree.child_04 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_04, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (thisOctree.child_05 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_05, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (thisOctree.child_06 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_06, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (thisOctree.child_07 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_07, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (thisOctree.child_08 > -1)
		{
			t = traverseOctree(r, octreees, thisOctree.child_08, geom, triangles, tmp_intersect, tmp_normal, t_min);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				intersectionPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}
		

		if (t_min > t_min_saved)
		{
			intersectionPoint = intersectionPoint_saved;
			normal = normal_saved;
			return t_min_saved;
		}
		else
		{
			return t_min;
		}


	}
	else
		return -1.0f;


}
*/





__host__ __device__ float traverseOctree(const Ray &r, const Octree *octreees, int thisOctreeNum, int OctreeSize, Geom geom,
	const Triangle* triangles, glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2 &uv, float t_min,
	int normalTexID, Image* ImageHeader, glm::vec3* imageData)
{
	float t;
	
	float minimum_t = t_min;


	bool bTraversed[MAX_OCTREE_CELL];

	for (int i = 0; i < MAX_OCTREE_CELL; i++)
		bTraversed[i] = false;
	
		
	bool bOutside;


	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	glm::vec2 tmp_uv;

	Octree thisOctree = octreees[thisOctreeNum]; //rootNode

	while (true)
	{
		if (BBIntersect(r, thisOctree.boundingBox, &t))
		{
			//child first
			if (thisOctree.child_01 > -1 && bTraversed[thisOctree.child_01] == false)
			{
				thisOctree = octreees[thisOctree.child_01];
			}
			else if (thisOctree.child_02 > -1 && bTraversed[thisOctree.child_02] == false)
			{
				thisOctree = octreees[thisOctree.child_02];
			}
			else if (thisOctree.child_03 > -1 && bTraversed[thisOctree.child_03] == false)
			{
				thisOctree = octreees[thisOctree.child_03];
			}
			else if (thisOctree.child_04 > -1 && bTraversed[thisOctree.child_04] == false)
			{
				thisOctree = octreees[thisOctree.child_04];
			}
			else if (thisOctree.child_05 > -1 && bTraversed[thisOctree.child_05] == false)
			{
				thisOctree = octreees[thisOctree.child_05];
			}
			else if (thisOctree.child_06 > -1 && bTraversed[thisOctree.child_06] == false)
			{
				thisOctree = octreees[thisOctree.child_06];
			}
			else if (thisOctree.child_07 > -1 && bTraversed[thisOctree.child_07] == false)
			{
				thisOctree = octreees[thisOctree.child_07];
			}
			else if (thisOctree.child_08 > -1 && bTraversed[thisOctree.child_08] == false)
			{
				thisOctree = octreees[thisOctree.child_08];
			}
			else
			{
				int index = thisOctree.firstElement;
				while (index > -1)
				{					
					t = triangleIntersectionTest(geom, triangles[index], r, tmp_intersect, tmp_normal, tmp_uv, bOutside, normalTexID, ImageHeader, imageData);

					if (t > 0.0f && minimum_t > t)
					{
						minimum_t = t;
						intersectionPoint = tmp_intersect;
						normal = tmp_normal;
						uv = tmp_uv;
					}

					index = triangles[index].nextTriangleID;
				}

				bTraversed[thisOctree.ID] = true;

				if (thisOctree.ParentID == -1)
					break;

				thisOctree = octreees[thisOctree.ParentID];

			}
		}
		else
		{
			bTraversed[thisOctree.ID] = true;

			if (thisOctree.ParentID == -1)
				break;

			thisOctree = octreees[thisOctree.ParentID];
		}
		
	}

	
	if (minimum_t >= t_min)
		return -1.0f;
	else
		return minimum_t;

}

__host__ __device__ Float SphericalTheta(const Vector3f &v)
{
	return glm::acos(glm::clamp(v.y, -1.0f, 1.0f));
}

__host__ __device__ Float SphericalPhi(const Vector3f &v)
{
	Float p = (glm::atan(v.z , v.x));
	
	
	//Float result = (p < 0.0f) ? (p + TwoPi) : p;

	return (p < 0.0f) ? (p + TwoPi) : p;
}

__host__ __device__ Color3f InfiniteAreaLight_L(const Vector3f &w, const Image  &ImageHeader, glm::vec3 *imageData)
{

	Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);

	//const Image &header = ImageHeader[EnvTexID];

	return getTextColor(ImageHeader.width, ImageHeader.height, ImageHeader.beginIndex, st, imageData);
}
