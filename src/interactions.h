#pragma once

#include "intersections.h"
#ifndef M_PI
	#define M_PI 3.14159265358979323846f
#endif

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

/** Square To Sphere Uniform Sample
*	This is used to wrap a square plane onto a sphere
*	Based on PBRT and CIS460
*/
__host__ __device__
glm::vec3 squareToSphereUniform(const glm::vec2 &sample)
{
	float z = 1.0f - 2.0f * sample[0];
	float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
	float phi = 2.0f * M_PI * sample[1];
	return glm::vec3(r * glm::cos(phi), r * std::sin(phi), z);
}

/** Sample Square
*/
__host__ __device__
glm::vec3 generateCubeSample(thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float x = u01(rng) - 0.5f;
	float y = u01(rng) - 0.5f;
	float z = u01(rng) - 0.5f;

	glm::vec3 sampledPoint(x, y, z);
	return sampledPoint;
}


/**
* Computes a cosine-weighted random direction in a hemisphere.
* Used for diffuse lighting.
* Implementation Reference: http://www.rorydriscoll.com/2009/01/07/better-sampling/
* Test this implementation for reference (The scene becomes darker? Why?)
*/
__host__ __device__
glm::vec3 CosineSampleHemisphere(thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float u1 = u01(rng);
	float u2 = u01(rng);
	const float r = std::sqrt(u1);
	const float theta = 2 * M_PI * u2;

	const float x = r * std::cos(theta);
	const float y = r * std::sin(theta);

	return glm::vec3(x, y, std::sqrt(std::max(0.0f, 1 - u1)));
}


// REMOVE BEFORE COMMIT
///**
//* Square to Disk Uniform mapping function
//* Based on the implementation in PBRT and CIS460
//*/
//__host__ __device__
//glm::vec2 squareToDiskUniform(const glm::vec2 point2D) {
//	float r = std::sqrt(point2D[0]); //x-axis maps to the radius
//	float theta = 2.0f * M_PI * point2D[1]; //y-axis maps to the angle on the disc
//	return glm::vec2(r * std::cos(theta), r * std::sin(theta));
//}

/**
* Square to Disk Concentric mapping function
* Based on the implementation in PBRT and CIS460
*/
__host__ __device__
glm::vec2 squareToDiskConcentric(const glm::vec2 point2D)
{
	float phi, r, u, v;
	float a = 2.0f * point2D[0] - 1.0f;
	float b = 2.0f * point2D[1] - 1.0f;


	if (a > -b)
	{
		if (a > b) //region 1
		{
			r = a;
			phi = (M_PI / 4.0f) * (b / a);
		}
		else // region 2
		{
			r = b;
			phi = (M_PI / 4.0f) * (2.0f - (a / b));
		}
	}
	else
	{
		if (a < b) // region 3
		{
			r = -a;
			phi = (M_PI / 4.0f) * (4.0f + (b / a));
		}
		else // region 4
		{
			r = -b;
			if (b != 0)
			{
				phi = (M_PI / 4.0f) * (6 - (a / b));
			}
			else
			{
				phi = 0;
			}
		}
	}

	u = r * std::cos(phi);
	v = r * std::sin(phi);
	return glm::vec2(u, v);
}

/**
*		FRESNEL
*		Reference: Scratchapixel: https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
*/
__host__ __device__
float fresnel(glm::vec3 incidentRay, glm::vec3 normal, const float ior) {
	float cosi = glm::clamp(glm::dot(incidentRay, normal), -1.f, 1.f);
	float etai = 1, etat = ior;
	if (cosi > 0) { std::swap(etai, etat); }
	// Compute sini using Snell's law
	float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
	// Total internal reflection
	if (sint >= 1) {
		return 1.f;
	}
	else {
		float cost = sqrtf(std::max(0.f, 1 - sint * sint));
		cosi = fabsf(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		return (Rs * Rs + Rp * Rp) / 2.f;
	}
}

/** REFRACTION
*	Finds the reflected or refracted ray
*/
__host__ __device__
glm::vec3 refract(thrust::default_random_engine &rng, glm::vec3 normal, glm::vec3 intersect, PathSegment& pathSegment, const Material& material, glm::vec3 incidentRay, bool outside) {
	glm::vec3 generatedRay;
	float ior = material.indexOfRefraction;
	float f = fresnel(incidentRay, normal, ior);
	
	thrust::uniform_real_distribution<float> u01(0, 1);
	if (f > u01(rng)) {
		generatedRay = glm::normalize(glm::reflect(incidentRay, normal));
	}
	else {
		float eta = (outside) ? 1.f / ior : ior;
		generatedRay = glm::normalize(glm::refract(incidentRay, normal, eta));
	}
	
	return generatedRay;
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
        thrust::default_random_engine &rng, 
		bool outside) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	// Implement the specular and the diffuse shading
	// Update the color of the sample
	// Generte new direction of the ray
	
	float probability = 1.0f; // Will be used later when slecting between refplection and refraction
	glm::vec3 newRayDirection;
	glm::vec3 finalColor = glm::vec3(1.0f,1.0f,1.0f);

	// SPECULAR REFLECTIVE
	if (m.hasReflective) {
		// Update direction 
		newRayDirection = glm::reflect(pathSegment.ray.direction , normal);

		// Update color
		finalColor = m.specular.color * m.color;
	}
	// TRANSMISSIVE
	else if (m.hasRefractive) {
		// Upadte direction
		newRayDirection = refract(rng, normal, intersect, pathSegment, m, pathSegment.ray.direction, outside);

		// Update color
		finalColor = m.color * m.specular.color;
	}
	// DIFFUSE
	else {
		// Update direction
		newRayDirection = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		float pdf = glm::dot(normal, newRayDirection) / M_PI;

		// Update color
		finalColor *= fabs(glm::dot(normal, newRayDirection)) * m.color / M_PI / pdf;
	}

	pathSegment.ray.direction = newRayDirection;
	pathSegment.ray.origin = intersect + 0.01f * newRayDirection;
	pathSegment.color *= finalColor;
}

/**
*		DEPTH OF FIELD
* This implementation assumes a thin lens approximation
* Based on the implementation in PBRT and CIS460
*/
__host__ __device__
void depthOfField(const Camera camera, thrust::default_random_engine& rng, PathSegment& segment) {
	if (camera.lRadius > 0) {
		// Generate a sample
		thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 point2D(u01(rng), u01(rng));
		point2D = squareToDiskConcentric(point2D);

		// Point on the lens
		glm::vec2 pLens = camera.lRadius * point2D;

		// Focal point
		glm::vec3 pFocus = segment.ray.origin + camera.fLength * segment.ray.direction;

		// Update the original ray from camera to start from the lens in the direction of the focal point
		glm::vec3 newOrigin = segment.ray.origin + (camera.up * pLens.y) + (camera.right * pLens.x);
		glm::vec3 newDirection = glm::normalize(pFocus - newOrigin);

		segment.ray.origin = newOrigin;
		segment.ray.direction = newDirection;
	}
	else {
		return;
	}
}

/**
*	DIRECT LIGHTING
*/
__host__ __device__
void ShadeDirectLighting(thrust::default_random_engine& rng, glm::vec3 intersectPoint, glm::vec3 surfaceNormal, Geom* geoms, int geoms_size, int* light_indixes, int no_of_lights, PathSegment& tempPS, glm::vec3 materialColor, Material* materials) {
	float t;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	thrust::uniform_real_distribution<float> u01(0, 1);

	// Randomly select one light and calculate a ray from the intersect point to the light based on the light polygon type sampling
	int light_index = u01(rng) * no_of_lights;
	Geom &lightGeom = geoms[light_indixes[light_index]];

	// Gnerate a point on the light then generate a "reflected ray" and pdf for the selected light from point of intersection to point on light
	glm::vec3 pointOnLight = squareToSphereUniform(glm::vec2(u01(rng), u01(rng)));
	glm::vec3 normalOnLight = glm::vec3(glm::normalize(lightGeom.inverseTransform * glm::vec4(pointOnLight, 0.0f)));
	// Transform the point on light with the transformation that are applied to the light
	pointOnLight = glm::vec3(lightGeom.transform * glm::vec4(pointOnLight, 1.0f));
	// Offsetting the point of intersection in order to avoid self interscetions
	pointOnLight = pointOnLight + EPSILON * normalOnLight;

	Ray r;
	r.direction = glm::normalize(pointOnLight - intersectPoint);
	r.origin = intersectPoint;
	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	bool outside = true;

	// Dot product between the reflected ray and the normal of the original point of intersection.
	float dotProd = glm::dot(r.direction, surfaceNormal);
	if (dotProd < 0.0f) {
		tempPS.color = glm::vec3(0.0f);
		return;
	}
	float adot = fabs(dotProd);
	float distance = glm::length(intersectPoint - pointOnLight);

	// Check if in shadow
	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];
		if (materials[geom.materialid].emittance > 0.0f) continue;

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
		}

		if (t + 0.0001f < distance && t != -1)
		{
			tempPS.color = glm::vec3(0.0f);
			return;
		}
	}

	// Divide the pdf by the number of lights
	float radius = lightGeom.scale.x;
	float pdf = distance * distance / (adot * radius * radius);
	pdf /= no_of_lights;

	// Handle edge case when pdf = 0
	if (pdf == 0.0f) {
		tempPS.color = glm::vec3(0.0f);
		return;
	}

	// Set the final color of the object
	tempPS.color *= materialColor * materials[geoms[light_index].materialid].color * materials[geoms[light_index].materialid].emittance * adot / pdf;
}