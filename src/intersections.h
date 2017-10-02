#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define CULL_BY_BBOX 1
#define PROCEDURAL_TEXTURE 0

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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside
#if PROCEDURAL_TEXTURE
        , glm::vec3 &uv
#endif
) {
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
#if PROCEDURAL_TEXTURE
        uv = getPointOnRay(q, tmin);
        intersectionPoint = multiplyMV(box.transform, glm::vec4(uv, 1.0f));
#else
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
#endif
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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside
#if PROCEDURAL_TEXTURE
  , glm::vec3 &uv
#endif
) {
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
#if PROCEDURAL_TEXTURE
    uv = objspaceIntersection;
#endif
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(Triangle tri, Geom geomMesh, Ray r,
  glm::vec3 &intersectionPoint, glm::vec3 &normal
#if PROCEDURAL_TEXTURE
  , glm::vec3 &uv
#endif
) {
  glm::vec3 bary;

  glm::vec3 v0 = tri.verts[0].pos;
  glm::vec3 v1 = tri.verts[1].pos;
  glm::vec3 v2 = tri.verts[2].pos;
  glm::vec3 fk = glm::vec3(geomMesh.transform * glm::vec4(0, 2, 2, 1));

  if (glm::intersectRayTriangle(r.origin, r.direction, v0, v1, v2, bary) || glm::intersectRayTriangle(r.origin, r.direction, v2, v1, v0, bary)) {

    float t = bary.z;
    intersectionPoint = getPointOnRay(r, t);
#if PROCEDURAL_TEXTURE
    uv = glm::vec3(bary.x, bary.y, 1 - bary.x - bary.z);
#endif
    normal = glm::normalize(bary.x * tri.verts[0].nor + bary.y * tri.verts[1].nor + (1.0f - bary.x - bary.y) * tri.verts[2].nor);//tri.verts[0].nor;
    if (glm::dot(normal, r.direction) > 0.0f) {
      normal *= -1.0f;
    }
    return t;
  }
  else {
    return -1.0f;
  }
}

__host__ __device__ float meshIntersectionTest(Geom meshGeom, Ray r, Mesh *meshes, Triangle *tris,
  glm::vec3 &intersectionPoint, glm::vec3 &normal
#if PROCEDURAL_TEXTURE
  , glm::vec3 &uv
#endif
) {

  Mesh mesh = meshes[meshGeom.meshIdx];
  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;
#if PROCEDURAL_TEXTURE
  glm::vec3 tmp_uv;
#endif
#if CULL_BY_BBOX
  // bbox test


  bool outside = true;

  Geom bbox;
  bbox.transform = mesh.bboxTransform;//glm::mat4(100.0f);
  bbox.inverseTransform = mesh.bboxInverseTransform; //glm::mat4(0.01f);
#if PROCEDURAL_TEXTURE
  if (boxIntersectionTest(bbox, r, tmp_intersect, tmp_normal, outside, tmp_uv) < 0.0f) {
#else
  if (boxIntersectionTest(bbox, r, tmp_intersect, tmp_normal, outside) < 0.0f) {
#endif
    return -1.0f;
  }
#endif
  // passed bbox test, check each tri
  float t_min = FLT_MAX;
  
  // I have no idea why, but a normal for-loop doesn't work
  // Instead, we start at start - 1 and index at i + 1 instead of i....???
  // indexing at i causes a single triangle to be rendered?? but not i + 1????
  for (int i = mesh.triangleStartIdx - 1; i < mesh.triangleEndIdx; i++) {
    Triangle tri = tris[i + 1];
#if PROCEDURAL_TEXTURE
    float t = triangleIntersectionTest(tri, meshGeom, r, tmp_intersect, tmp_normal, tmp_uv);
#else
    float t = triangleIntersectionTest(tri, meshGeom, r, tmp_intersect, tmp_normal);
#endif

    if (t >= 0.0f && t < t_min) {
      t_min = t;
      intersectionPoint = tmp_intersect;
      normal = tmp_normal;
#if PROCEDURAL_TEXTURE
      uv = tmp_uv;
#endif
    }
  }
  return (t_min == FLT_MAX) ? -1.0f : t_min;
}