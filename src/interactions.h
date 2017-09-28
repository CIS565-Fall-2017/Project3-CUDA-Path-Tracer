#pragma once

#include "intersections.h"
#include "build\src\BSDF.h"


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

__host__ __device__ glm::vec3 squareToDiskConcentric(glm::vec2 sample)
{
	float phi, r, u, v;
	float a = 2 * sample[0] - 1;
	float b = 2 * sample[1] - 1;

	if (a>-b)
	{
		if (a>b)
		{
			r = a;
			phi = (PI / 4)*(b / a);
		}
		else
		{
			r = b;
			phi = (PI / 4)*(2 - (a / b));
		}
	}
	else
	{
		if (a<b)
		{
			r = -a;
			phi = (PI / 4)*(4 + (b / a));
		}
		else
		{
			r = -b;
			if (b != 0)
			{
				phi = (PI / 4)*(6 - (a / b));
			}
			else
			{
				phi = 0;
			}
		}
	}

	u = r*cos(phi);
	v = r*sin(phi);
	return glm::vec3(u, v, 0);
}

__host__ __device__ glm::vec3 Clamp(glm::vec3 vec, float small, float large)
{
	glm::vec3 temp;
	if (vec.x > large)
	{
		temp.x = large;
	}
	if (vec.x < small)
	{
		temp.x = small;
	}
	if (vec.y > large)
	{
		temp.y = large;
	}
	if (vec.y < small)
	{
		temp.y = small;
	}
	if (vec.z > large)
	{
		temp.z = large;
	}
	if (vec.z < small)
	{
		temp.z = small;
	}
	return temp;
}

__host__ __device__ float TangentSpaceZ(glm::vec3 normal, glm::vec3 rayDir)
{
	
	return glm::dot(normal,rayDir)/glm::length(normal);
}

__host__ __device__ glm::vec3 TangentSpaceToWorldSpace(glm::vec3 normal, glm::vec3 worldVec)
{
	return glm::vec3(normal.x*worldVec.z, normal.y*worldVec.z, normal.z*worldVec.z);
}

__host__ __device__ glm::vec3 CosineWeightedRandomSample(
	glm::vec3 normal, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::uniform_real_distribution<float> u02(0, 1);

	glm::vec2 sample = glm::vec2(u01(rng), u02(rng));

	glm::vec3 flatHemisphere = squareToDiskConcentric(sample);
	float z_coordinate = std::sqrt(1 - flatHemisphere[0] * flatHemisphere[0] - flatHemisphere[1] * flatHemisphere[1]);

	return TangentSpaceToWorldSpace(normal, glm::vec3(flatHemisphere[0], flatHemisphere[1], z_coordinate));
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
	if (pathSegment.remainingBounces <= 0)
	{
		return;
	}

	//terminate the black rays	

	thrust::uniform_real_distribution<float> u01(0, 1);
	
	glm::vec3 newRayDir;
	glm::vec3 lastRayDir = pathSegment.ray.direction;
	float pdf;
	glm::vec3 fColor;
	glm::vec3 finalColor(0.0f);
	glm::vec3 lastColor = pathSegment.color;

	//lambert
	if ((m.hasReflective==0.f)&&(m.hasRefractive==0.f))
	{
		newRayDir = calculateRandomDirectionInHemisphere(normal, rng);
		//fColor = m.color / PI;
		//float zValue = TangentSpaceZ(normal,newRayDir);
		//pdf = zValue / PI;
 
		//if (pdf > 0.f)
		//{
			//finalColor = fColor*lastColor*abs(glm::dot(newRayDir, normal) / pdf);
		finalColor = m.color*lastColor;//*abs(glm::dot(newRayDir, normal));
		//}
		//else
		//{
		//	finalColor = m.color*lastColor;
		//}
	}
	//FresnelDielectric
	//else if((m.hasReflective>0.f)&&(m.hasRefractive>0.f))
	//{
	//	FresnelDielectric fresnel = FresnelDielectric(m.hasReflective, m.hasRefractive);
	//	float randomResult = u01(rng);

	//	if (randomResult > 0.5)
	//	{
	//		SpecularBRDF brdf = SpecularBRDF(m.specular.color, &fresnel);
	//		fColor = brdf.Sample_f(lastRayDir, &newRayDir, &pdf);
	//		finalColor = fColor*lastColor*abs(glm::dot(newRayDir, normal) / pdf);
	//	}
	//	else
	//	{
	//		SpecularBTDF btdf = SpecularBTDF(m.specular.color, m.hasReflective, m.hasRefractive, &fresnel);
	//		fColor = btdf.Sample_f(lastRayDir, &newRayDir, &pdf);
	//		finalColor = fColor*lastColor*abs(glm::dot(newRayDir, normal) / pdf);
	//	}
	//}
	//perfect specular
	else if (m.specular.exponent==1.0f)
	{
		newRayDir = glm::reflect(pathSegment.ray.direction, normal);
		finalColor = lastColor * m.specular.color;

	}
	//other situations 
	else
	{
		float randomResult = u01(rng);

		//lambert
		if (randomResult >= 0.5)
		{
			newRayDir = calculateRandomDirectionInHemisphere(normal, rng);

			finalColor = m.color*lastColor*glm::vec3(0.7f);
		}
		//specular
		else
		{
			newRayDir = glm::reflect(pathSegment.ray.direction, normal);
			finalColor = lastColor * m.specular.color*glm::vec3(0.3f);
		}
	}

	//finalColor = Clamp(finalColor, 0.f, 1.0f);
	finalColor = glm::clamp(finalColor, 0.f, 1.f);
	pathSegment.ray.origin = intersect +newRayDir*FLT_EPSILON;
	pathSegment.ray.direction = newRayDir;

	pathSegment.color = finalColor;
}
