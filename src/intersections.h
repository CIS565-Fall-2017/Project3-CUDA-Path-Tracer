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

__host__ __device__ float triangleIntersectionTest(Triangle tri, Geom geomMesh, Ray r,
  glm::vec3 &intersectionPoint, glm::vec3 &normal) {
  glm::vec3 bary;
#if 0
  if (abs(tri.verts[0].pos.x - 0) > 0.0001f) {
    cout << "x " << endl;
  }
  if (abs(tri.verts[0].pos.y - 2) > 0.0001f) {
    cout << "y " << endl;
  }
  if (abs(tri.verts[0].pos.z - 2) > 0.0001f) {
    cout << "z " << endl;
  }
#endif
  glm::vec3 v0 = tri.verts[0].pos;
  glm::vec3 v1 = tri.verts[1].pos;
  glm::vec3 v2 = tri.verts[2].pos;
  glm::vec3 fk = glm::vec3(geomMesh.transform * glm::vec4(0, 2, 2, 1));
#if 0
  if (abs(v0.x - fk.x) > 0.0001f) {
    cout << "XX " << endl;
  }
  if (abs(v0.y - fk.y) > 0.0001f) {
    cout << "YY " << endl;
  }
  if (abs(v0.z - fk.z) > 0.0001f) {
    cout << "ZZ " << endl;
  }
#endif
  if (glm::intersectRayTriangle(r.origin, r.direction, v0, v1, v2, bary) || glm::intersectRayTriangle(r.origin, r.direction, v2, v1, v0, bary)) {
#if 0
                                glm::vec3(geomMesh.transform * glm::vec4(0, 2, 2, 1)),
                                glm::vec3(geomMesh.transform * glm::vec4(0, 0, 2, 1)),
                                glm::vec3(geomMesh.transform * glm::vec4(2, 0, 2, 1)), bary)) {
#endif
    float t = bary.z;
    intersectionPoint = getPointOnRay(r, t);
    normal = tri.verts[0].nor;
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
  glm::vec3 &intersectionPoint, glm::vec3 &normal) {
  // bbox test
  Mesh mesh = meshes[meshGeom.meshIdx];
  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;
  bool outside = true;

  Geom bbox;
  bbox.transform = mesh.bboxTransform;//glm::mat4(100.0f);
  bbox.transform[3][3] = 1.0f;
  bbox.inverseTransform = mesh.bboxInverseTransform; //glm::mat4(0.01f);
  bbox.inverseTransform[3][3] = 1.0f;
  if (boxIntersectionTest(bbox, r, tmp_intersect, tmp_normal, outside) < 0.0f) {
    return -1;
  }

  // passed bbox test, check each tri
  float t_min = FLT_MAX;

  
  for (int i = mesh.triangleStartIdx; i < mesh.triangleEndIdx; i++) {
    Triangle tri = tris[i + 1];
    float t = triangleIntersectionTest(tri, meshGeom, r, tmp_intersect, tmp_normal);
    if (t >= 0.0f && t < t_min) {
      t_min = t;
      intersectionPoint = tmp_intersect;
      normal = tmp_normal;
    }
  }
  return (t_min == FLT_MAX) ? -1.0f : t_min;
}