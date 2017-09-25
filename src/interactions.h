#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// squareToDiskConcentric sampling from CIS 561
__host__ __device__
glm::vec2 squareToDiskConcentric(const glm::vec2 &sample)
{
  float phi, r, u, v;
  float a = 2.0f * sample[0] - 1.0f; // a and b are in range [-1,1]
  float b = 2.0f * sample[1] - 1.0f;
  if (a > -b) // region 1 or 2
  {
    if (a > b) // region 1, also |a| > |b|
    {
      r = a;
      phi = PiOver4 * (b / a);
    }
    else // region 2, also |b| > |a|
    {
      r = b;
      phi = PiOver4 * (2.0f - (a / b));
    }
  }
  else // region 3 or 4
  {
    if (a < b) // region 3, also |a| >= |b|, a != 0
    {
      r = -a;
      phi = PiOver4 * (4.0f + (b / a));
    }
    else // region 4, |b| >= |a|, but a==0 and b==0 could occur.
    {
      r = -b;
      if (b != 0.0f)
        phi = PiOver4 * (6.0f - (a / b));
      else
        phi = 0.0f;
    }
  }
  u = r*std::cos(phi);
  v = r*std::sin(phi);
  return glm::vec2(u, v);
}

// PBRT like Thin Lens Camera..
__host__ __device__
void applyDof(Ray &ray, thrust::default_random_engine &rng, Camera cam)
{
  if (cam.lensRadius > 0)
  {
    //    Generate Sample
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 p(u01(rng),u01(rng));
   
    // FIX THIS
   /* WarpFunctions wf;
    p = Point2f(wf.squareToDiskConcentric(p));*/
    p = squareToDiskConcentric(p);

    //    scale to get a point on lens
    glm::vec2 pLens = cam.lensRadius * p;

    //    Compute point on plane of focus
    glm::vec3 pFocus = cam.focalLength * ray.direction + ray.origin;

    //    Update ray for effect of lens
    ray.origin = ray.origin + (cam.up*pLens.y) + (cam.right*pLens.x);
    ray.direction = glm::normalize(pFocus - ray.origin);
  }
}



/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

  glm::vec3 col(1.f);
  glm::vec3 dir;

  if (m.hasReflective) {
    dir = glm::reflect(pathSegment.ray.direction, normal); // pure specular reflection
    col *= m.specular.color * m.color;
    pathSegment.ray.origin = intersect + normal * EPSILON; // double check this..
  }
  else if (m.hasRefractive) {
    // ..
        // TODO: replace with Schlick's approximation or fresnel dielectric..
    thrust::uniform_real_distribution<float> u01(0, 1);
    if (u01(rng) > 0.5f) {
      // DIFFUSE and specular split..
          dir = calculateRandomDirectionInHemisphere(normal, rng);
          col *= m.color * 2.f;
      // REFRACTIVE
          //float ior = pathSegment.refractEnter ? m.indexOfRefraction : 1.f / m.indexOfRefraction;
          //dir = glm::refract(pathSegment.ray.direction, normal, ior);
          //col *= m.color * 1.f;

          //glm::vec3 nor = pathSegment.refractEnter ? normal : -normal;
          pathSegment.ray.origin = intersect + normal * EPSILON; // double check this..

          //pathSegment.refractEnter = !pathSegment.refractEnter;
    }
    else {
      dir = glm::reflect(pathSegment.ray.direction, normal); // pure specular reflection
      col *= m.specular.color * 2.f;
      pathSegment.ray.origin = intersect + normal * EPSILON; // double check this..
    }
  }
  else {
    dir = calculateRandomDirectionInHemisphere(normal, rng);
    col *= m.color;//*InvPI;
    pathSegment.ray.origin = intersect + normal * EPSILON; // double check this..
  }

  pathSegment.color *= fabs(glm::dot(normal, dir)) * col;
  pathSegment.ray.direction = glm::normalize(dir);
  //pathSegment.ray.origin = intersect + normal * EPSILON; // double check this..
}
