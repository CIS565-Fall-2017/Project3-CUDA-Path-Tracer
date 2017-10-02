#pragma once

#include "intersections.h"

#define FRESNEL 0

/***********************/
/******* SAMPLING ******/
/***********************/

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

__host__ __device__
glm::vec3 sampleSphereUniform(thrust::default_random_engine &rng)
{
  thrust::uniform_real_distribution<float> u01(0, 1);
  glm::vec2 sample(u01(rng), u01(rng));
  float z = 1.0f - 2.0f * sample[0];
  float r = sqrt(max(0.0f, 1.0f - z * z));
  float phi = TWO_PI * sample[1];
  return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
}

// Not sure about this one..
__host__ __device__
glm::vec3 sampleCubeUniform(thrust::default_random_engine &rng)
{
  thrust::uniform_real_distribution<float> u55(-0.5, 0.5);
  int fixedAxis = 3.f * (u55(rng) + 0.5f); // choose a fix axis
  float val = u55(rng) > 0 ? 0.5 : -0.5;

  if (fixedAxis == 0) {
    return glm::vec3(val, u55(rng), u55(rng));
  }
  else if (fixedAxis == 1) {
    return glm::vec3(u55(rng), val, u55(rng));
  }
  else {
    return glm::vec3(u55(rng), u55(rng), val);
  }
}


/************************/
/**** DEPTH OF FIELD ****/
/************************/

// PBRT like Thin Lens Camera..
__host__ __device__
void applyDof(Ray &ray, thrust::default_random_engine &rng, Camera cam)
{
  // Generate Sample
  thrust::uniform_real_distribution<float> u01(0, 1);
  glm::vec2 p(u01(rng),u01(rng));
   
  // Sample on a disk
  p = squareToDiskConcentric(p);

  // Scale to get a point on lens
  glm::vec2 pLens = cam.lensRadius * p;

  // Compute point on plane of focus
  glm::vec3 pFocus = cam.focalLength * ray.direction + ray.origin;

  // Update ray for effect of lens
  ray.origin = ray.origin + (cam.up*pLens.y) + (cam.right*pLens.x);
  ray.direction = glm::normalize(pFocus - ray.origin);
}

/***********************/
/******* SHADING *******/
/***********************/

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
glm::vec3 refract(glm::vec3 normal, glm::vec3 ray, float n, bool outside, thrust::default_random_engine &rng)
{
  thrust::uniform_real_distribution<float> u01(0, 1);

#if !FRESNEL // SCHLICK'S APPROX
  float n2 = outside ? 1.f / n : n;
  float R0 = powf((1.0f - n2) / (1.0f + n2), 2.f);
  float c = powf(1.0f - abs(glm::dot(ray, normal)), 5);
  float RTheta = R0 + (1.f - R0) * powf(1.f - fabs(glm::dot(ray, normal)), 5.f);
  if (RTheta < u01(rng)) {
    return glm::normalize(glm::refract(ray, normal, n2));
  }
  else {
    return glm::normalize(glm::reflect(ray, normal));
  }

#else // FRESNEL

  float cosThetaI = glm::dot(ray, normal);
  float c_cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

  bool entering = c_cosThetaI > 0.f;
  float eI = 1.f, eT = n;
  if (!entering) {
    std::swap(eI, eT);
    c_cosThetaI = fabs(c_cosThetaI);
  }

  float sinThetaI = sqrtf(max(0.f, 1.f - c_cosThetaI * c_cosThetaI));
  float sinThetaT = eI / eT * sinThetaI;

  if (sinThetaT >= 1.f)
    return glm::normalize(glm::reflect(ray, normal));

  float cosThetaT = sqrtf(max(0.f, 1.f - sinThetaT * sinThetaT));
  float Rparl = ((eT * c_cosThetaI) - (eI * cosThetaT)) / ((eT * c_cosThetaI) + (eI * cosThetaT));
  float Rperp = ((eI * c_cosThetaI) - (eT * cosThetaT)) / ((eI * c_cosThetaI) + (eT * cosThetaT));
  float f = ((Rparl * Rparl + Rperp * Rperp) / 2.f);

  if (f > u01(rng)) {
    return glm::normalize(glm::reflect(ray, normal));
  }
  else {
    float n = outside ? 1.f / n : n;
    return glm::normalize(glm::refract(ray, normal, n));
  }

#endif
}

// HEURISTIC FOR MIS
__host__ __device__
float powerHeuristic(float f, float g)
{
  //TODO
  float denom = (f*f + g*g);
  if (denom == 0.f)
    return 0.f;
  return (f*f) / denom;
}

// NAIVE INTEGRATION
__host__ __device__
float scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        bool outside,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

  glm::vec3 dir;
  glm::vec3 col;
  float absDot, pdf, dot;

  if (m.hasReflective) {
    dir = glm::reflect(pathSegment.ray.direction, normal); // pure specular reflection
    absDot = 1.f;
    pdf = 1.f;
    pathSegment.color *= m.color * m.specular.color;
  }
  else if (m.hasRefractive) {
    dir = refract(normal, pathSegment.ray.direction, m.indexOfRefraction, outside, rng);
    absDot = 1.f;
    pdf = 1.f;
    pathSegment.color = m.color * m.specular.color;
  }
  else {
    dir = calculateRandomDirectionInHemisphere(normal, rng);
    float cosTheta = glm::dot(normal, -pathSegment.ray.direction);
    absDot = max(0.f, cosTheta);
    pdf = cosTheta / PI;
    pathSegment.color *= m.color / PI * 
                          absDot / 
                          pdf;
  }

  pathSegment.ray.direction = glm::normalize(dir);
  pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction; //normal * EPSILON; // double check this..
  return pdf;
}



/***********************/
/*** DIRECT LIGHTING ***/
/***********************/

// PURE DIRECT LIGHT
__host__ __device__
float computeDirectLight(
  PathSegment & pathSegment, glm::vec3 intersect, glm::vec3 normal,
  const Material &m, const Material * materials,
  thrust::default_random_engine &rng,
  Geom * geoms, Geom * lights, int numGeoms, int numLights) 
{
  // DIRECT LIGHTING SHADING COMPUTATION
  // 1. generate light sample
  // 2. check for occlusion
  // 3. add contribution

  // RETURN BLACK FOR SPECULAR???
  if (m.hasReflective == 1.0 || m.hasRefractive == 1.0) {
    pathSegment.color *= 0.f;
    return 0.f;
  }

  thrust::uniform_real_distribution<float> u01(0, 1);

  // choose light...
  int lrand = numLights * u01(rng);
  const Geom &light = lights[lrand];
  float pdf;

  // sample light...
  glm::vec3 sample;
  if (light.type == CUBE) {
    sample = sampleCubeUniform(rng);
  }
  else if (light.type == SPHERE) {
    sample = sampleSphereUniform(rng);
  }
  else return 0.f; // INVALID CASE

  // return if direction is not proper..
  sample = glm::vec3(light.transform * glm::vec4(sample, 1.0f));
  glm::vec3 rayToSample = sample - intersect;
  float dotProd = glm::dot(glm::normalize(rayToSample), normal);

  float absDot = fabs(dotProd);
  if (dotProd < 0.0) {
    pathSegment.color *= 0.f;
    return 0.f;
  }
  
  float sample_t = glm::length(rayToSample);

  // detect occlusion...
  PathSegment ps = pathSegment;
  ps.ray.origin = intersect + EPSILON * normal;
  ps.ray.direction = glm::normalize(rayToSample);
  float t;
  bool outside = true;
  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;

  for (int i = 0; i < numGeoms; i++) {
    Geom & geom = geoms[i];
    //if (geom.materialid == light.materialid) continue;

    if (geom.type == CUBE) {
      t = boxIntersectionTest(geom, ps.ray, tmp_intersect, tmp_normal, outside);
    }
    else if (geom.type == SPHERE) {
      t = sphereIntersectionTest(geom, ps.ray, tmp_intersect, tmp_normal, outside);
    }

    // compare with light intersection point's t to detect occlusion
    if (t + 0.001f < sample_t && t != -1 && geom.materialid != light.materialid) {
      pathSegment.color *= 0.f;
      return 0.f;
    }
  }

  if (light.type == SPHERE) {
    pdf = sample_t * sample_t
         // / absDot   // absdot is 1 for sphere lights..
          / (light.scale.x * light.scale.x);
  }

  pdf /= numLights;
  if (pdf != 0) {
    // diffuse only..
    pathSegment.color = m.color * absDot / pdf *
      materials[light.materialid].color * materials[light.materialid].emittance;
  }

  return pdf;
}



/*******************/
/******* MIS *******/
/*******************/


// SAMPLE SPHERE LIGHT
__host__ __device__
float pdfSphereLi(const Geom & light, PathSegment ps, 
    glm::vec3 intersect, glm::vec3 normal, glm::vec3 rayToSample) 
{
  //PathSegment ps = pathSegment;
  ps.ray.origin = intersect + EPSILON * normal;
  ps.ray.direction = glm::normalize(rayToSample);

  float t;
  bool outside = true;
  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;

  if (light.type == CUBE) {
    t = boxIntersectionTest(light, ps.ray, tmp_intersect, tmp_normal, outside);
  }
  else if (light.type == SPHERE) {
    t = sphereIntersectionTest(light, ps.ray, tmp_intersect, tmp_normal, outside);
  }

  if (t == -1.f) {
    return 0.f;
  }

  glm::vec3 dist = intersect - tmp_intersect;
  float absDot = fabs(glm::dot(tmp_normal, glm::normalize(dist))); // should be 1 for sphere lights..

  float pdf;
  if (light.type == SPHERE && absDot !=0) {
    pdf = t * t
      / (/*absDot*/ 1.f * 4 * PI * light.scale.x * light.scale.x);
  }

  return pdf;
}

__host__ __device__
float pdfLambert(const glm::vec3 &dir, const glm::vec3 &normal)
{
  //TODO
  //    if(!SameHemisphere(wo,wi))
  //        return 0.f;
  float costheta = glm::dot(dir, normal);
  if (costheta < 0.0f) {
    return 0.f;
  }
  return costheta / PI;
}

// LIGHT SAMPLE FOR FULL LIGHT INTEGRATOR
__host__ __device__
glm::vec3 sampleLightMIS(
  PathSegment & pathSegment, glm::vec3 intersect, glm::vec3 normal,
  const Material &m, const Material * materials,
  thrust::default_random_engine &rng,
  const Geom * geoms, const Geom * lights, int numGeoms, int numLights)
{
  // RETURN BLACK FOR SPECULAR???
  if (m.hasReflective || m.hasRefractive) {
    //pathSegment.color *= 0.f;
    return glm::vec3(0.f);
  }

  thrust::uniform_real_distribution<float> u01(0, 1);

  // choose light...
  int lrand = numLights * u01(rng);
  const Geom &light = lights[lrand];
  float pdf, fpdf, gpdf;

  // sample light...
  glm::vec3 sample;
  if (light.type == CUBE) {
    sample = sampleCubeUniform(rng);
  }
  else if (light.type == SPHERE) {
    sample = sampleSphereUniform(rng);
  }
  else return glm::vec3(0.f); // INVALID CASE

  // return if direction is not proper..
  sample = glm::vec3(light.transform * glm::vec4(sample, 1.0f));
  glm::vec3 rayToSample = sample - intersect;
  float dotProd = glm::dot(glm::normalize(rayToSample), normal);

  glm::vec3 col, colL, colB;

  float absDot = fabs(dotProd);
  if (dotProd < 0.0) {
    return glm::vec3(0.f);
  }

  float sample_t = glm::length(rayToSample);

  // detect occlusion...
  PathSegment ps = pathSegment;
  ps.ray.origin = intersect + EPSILON * normal;
  ps.ray.direction = glm::normalize(rayToSample);
  float t;
  bool outside = true;
  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;

  for (int i = 0; i < numGeoms; i++) {
    const Geom & geom = geoms[i];
    //if (geom.materialid == light.materialid) continue;

    if (geom.type == CUBE) {
      t = boxIntersectionTest(geom, ps.ray, tmp_intersect, tmp_normal, outside);
    }
    else if (geom.type == SPHERE) {
      t = sphereIntersectionTest(geom, ps.ray, tmp_intersect, tmp_normal, outside);
    }

    // compare with light intersection point's t to detect occlusion
    if (t + 0.001f < sample_t && t != -1 && geom.materialid != light.materialid) {
      return glm::vec3(0.f);
    }
  }

  if (light.type == SPHERE) {
    pdf = sample_t * sample_t
      // / absDot // absdot is 1 for sphere lights
      / (light.scale.x * light.scale.x);
  }

  gpdf = pdf;
  //pdf /= numLights;
  if (gpdf != 0) {
    // diffuse only..
    fpdf = pdfLambert(pathSegment.ray.direction, glm::normalize(rayToSample));
    col = m.color * absDot / gpdf;
    col *= materials[light.materialid].color * materials[light.materialid].emittance;
    col *= (1 - powerHeuristic(fpdf, gpdf));
    colL = col;

   /* col = m.color * absDot / pdf *
      materials[light.materialid].color * materials[light.materialid].emittance;
    colL = col;*/
  }
  else return glm::vec3(0.f);

  // SAMPLE BSDF FOR MIS

  glm::vec3 dir;
  col = glm::vec3(0.f);
  absDot = dotProd = pdf = fpdf = gpdf = 0;

  // only diffuse surfaces..
  dir = calculateRandomDirectionInHemisphere(normal, rng);
  dotProd = glm::dot(normal, dir);
  if (dotProd != 0.f) { // do this only if fpdf!=0..
    absDot = max(0.f, dotProd);
    pdf = dotProd / PI;
    /*col = m.color / PI *
      absDot /
      pdf;*/
  }

  fpdf = pdf;
  if (fpdf != 0.f) {

    dir = glm::normalize(dir);
    glm::vec3 origin = intersect + 0.01f * dir;

    // detect occlusion...
    PathSegment ps = pathSegment;
    ps.ray.origin = origin + EPSILON * normal;
    ps.ray.direction = dir;
    float t;
    bool outside = true;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    bool hit = true;

    int ind = 0;

    for (int i = 0; i < numGeoms; i++) {
      const Geom & geom = geoms[i];
      //if (geom.materialid == light.materialid) continue;

      if (geom.type == CUBE) {
        t = boxIntersectionTest(geom, ps.ray, tmp_intersect, tmp_normal, outside);
      }
      else if (geom.type == SPHERE) {
        t = sphereIntersectionTest(geom, ps.ray, tmp_intersect, tmp_normal, outside);
      }

      // compare with light intersection point's t to detect occlusion
      if (t + 0.001f < sample_t && t != -1 && geom.materialid != light.materialid) {
        hit = false;
      }
    }

    if (hit) {
      gpdf = pdfSphereLi(light, pathSegment, intersect, normal, pathSegment.ray.direction) / numLights;
      col = m.color / PI *
        absDot /
        pdf;
      col += materials[light.materialid].color * materials[light.materialid].emittance;
      col *= powerHeuristic(fpdf, gpdf);
      colB = col;
    }
  }

  return float(numLights) * (colL + colB);
}


// BSDF SAMPLE FOR FULL LIGHT INTEGRATOR
__host__ __device__
glm::vec3 sampleBsdfMIS(
  PathSegment & pathSegment, glm::vec3 intersect, glm::vec3 normal, bool outside, 
  const Material &m, thrust::default_random_engine &rng,
  const Geom * lights, int numLights)
{
  // TODO
  glm::vec3 dir;
  glm::vec3 col(0.f);
  float absDot, pdf, dot;
  float fpdf, gpdf;

  if (m.hasReflective) {
    dir = glm::reflect(pathSegment.ray.direction, normal); // pure specular reflection
    absDot = 1.f;
    pdf = 1.f;
    col = m.color * m.specular.color;
  }
  else if (m.hasRefractive) {
    dir = refract(normal, pathSegment.ray.direction, m.indexOfRefraction, outside, rng);
    absDot = 1.f;
    pdf = 1.f;
    col = m.color * m.specular.color;
  }
  else {
    dir = calculateRandomDirectionInHemisphere(normal, rng);
    float cosTheta = glm::dot(normal, pathSegment.ray.direction);
    if (cosTheta != 0.f) { // do this only if fpdf!=0..
      absDot = max(0.f, cosTheta);
      pdf = cosTheta / PI;
      col = m.color / PI *
        absDot /
        pdf;
    }
  }

  pathSegment.ray.direction = glm::normalize(dir);
  pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;

  fpdf = pdf;

  //// LIGHT PDF FOR MIS
  thrust::uniform_real_distribution<float> u01(0, 1);
  // choose light...
  int lrand = numLights * u01(rng);
  const Geom &light = lights[lrand];
  gpdf = pdfSphereLi(light, pathSegment, intersect, normal, pathSegment.ray.direction) / numLights; // TODO
  col *= powerHeuristic(fpdf, gpdf);

  return col;
}

