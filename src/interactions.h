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
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
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
/*
__host__ __device__
void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    float& pdf) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    switch (m.bxdf) {
    case EMISSIVE:
        pdf = 1.f;
        break;
    case DIFFUSE:
        // Cosine-weighted hemisphere sampling
        pathSegment.ray.origin = (glm::dot(normal, pathSegment.ray.direction) > 0) ? intersect + normal * EPSILON : intersect + normal * -EPSILON;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pdf = glm::dot(glm::normalize(pathSegment.ray.direction), glm::normalize(normal)) / PI;
        break;
    case SPECULAR_BRDF:
        pathSegment.ray.origin = (glm::dot(normal, pathSegment.ray.direction) > 0) ? intersect + normal * EPSILON : intersect + normal * -EPSILON;
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pdf = 1.f; // delta distribution
        break;
    case SPECULAR_BTDF:
        // refract stuff? snell's law. also, need dielectric fresnel
        break;
    }
}
*/

__device__
float PowerHeuristic(float nf, float fPdf, float ng, float gPdf)
{
    if (fPdf == 0.0f)
    {
        return 0.0f;
    }

    // Reasoning for beta = 2 can be found in Veach's thesis: https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf
    const float beta = 2.f;
    float powFPdf = pow((float)nf * fPdf, beta);
    return (powFPdf) / (powFPdf + pow((float)ng * gPdf, beta));
}

__device__
float ComputeLightPDF(
    Geom & light
    , glm::vec3 lightNormal
    , Ray r // ray coming from the geometry being shaded
)
{
    bool outside;
    glm::vec3 lightIsect;
    float t = planeIntersectionTest(light, r, lightIsect, lightNormal, outside);
    if (t <= 0.0f)
    {
        return 0.0f;
    }
    else
    {
        // Compute solid angle pdf
        glm::vec3 diffVector = lightIsect - r.origin;
        float distanceSq = glm::length2(diffVector);
        diffVector = glm::normalize(diffVector);

        float absDot = abs(glm::dot(-diffVector, lightNormal));
        if (absDot <= EPSILON)
        {
            return 0.0f;
        }
        else
        {
            glm::vec3 lightScale = light.scale;
            float area = lightScale.x * lightScale.y; // compute area of the light
            return 1.0f / area * distanceSq / absDot;
        }
    }
}
