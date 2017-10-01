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

#define INVPI 0.31830988618379067154f

__host__ __device__
bool SameHemisphere(glm::vec3 &w, glm::vec3 &wp)
{
	return w.z * wp.z > 0;
}

__host__ __device__ float CosTheta(const glm::vec3 wi,  const glm::vec3 n) {
	return glm::dot(n, wi);
}

__host__ __device__ float AbsCosTheta(glm::vec3 wi, glm::vec3 n) {
	return glm::abs(CosTheta(wi, n));
}
__host__ __device__
float getPdf(glm::vec3 &wo, glm::vec3 &wi, glm::vec3 &n)
{
	return SameHemisphere(wo, wi) ? AbsCosTheta(wi, n) * INVPI : 0;
}

__host__ __device__
float AbsDot(const glm::vec3 n, const glm::vec3 wi)
{
	return glm::abs(glm::dot(n, wi));
}

__host__ __device__
glm::vec3 fresnelDielectric(glm::vec3 &wo, glm::vec3 &wi, glm::vec3 &normal, float etaI, float etaT)
{
	float cosThetaI = glm::clamp(CosTheta(wi, normal), -1.f, 1.f);

	bool entering = cosThetaI > 0.f;
	if (!entering) {
		float temp = etaI;
		etaI = etaT;
		etaT = temp;

		cosThetaI = glm::abs(cosThetaI);
	}

	// Snell's law
	float sinThetaI = glm::sqrt(glm::max(0.f, 1 - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;

	// Total internal reflection
	if (sinThetaT >= 1.f) {
		return glm::vec3(1.f);
	}

	float cosThetaT = glm::sqrt(glm::max(0.f, 1 - sinThetaT * sinThetaT));

	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

	float fr = Rparl * Rparl;
	fr += Rperp * Rperp;
	fr /= 2.f;

	return glm::vec3(fr);
}

__host__ __device__
bool Refract(glm::vec3 &wi, glm::vec3 &n, float eta, glm::vec3 &wt)
{
	// Compute cos theta using Snell's law
	float cosThetaI = glm::dot(n, wi);
	float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = std::sqrt(1 - sin2ThetaT);
	wt = eta * -wi + (eta * cosThetaI - cosThetaT) * glm::vec3(n);

	return true;
}

__host__ __device__
glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v)
{
	return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__
float SinTheta2(const glm::vec3 &w, const glm::vec3 &normal)
{
	return max(0.f, 1.f - CosTheta(w, normal) * CosTheta(w, normal));
}

__host__ __device__
void spawnRay(PathSegment &pathSegment, const glm::vec3 &normal, const glm::vec3 &wi, const glm::vec3 &intersect)
{
	glm::vec3 originOffset = normal * EPSILON;
	originOffset = (glm::dot(wi, normal) > 0) ? originOffset : -originOffset;
	pathSegment.ray.origin = intersect + originOffset;
	pathSegment.ray.direction = wi;
}

__host__ __device__
void IntersectRay(
	PathSegment &pathSegment,
	Geom *geoms,
	int geoms_size,
	ShadeableIntersection &intersection)
{
	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// naive parse through global geoms

	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		// TODO: add more intersection tests here... triangle? metaball? CSG?

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
			intersect_point = tmp_intersect;
			normal = tmp_normal;
		}
	}

	if (hit_geom_index == -1)
	{
		intersection.t = -1.0f;
	}
	else
	{
		//The ray hits something
		intersection.t = t_min;
		intersection.materialId = geoms[hit_geom_index].materialid;
		intersection.surfaceNormal = normal;
		intersection.intersectPoint = intersect_point;
	}
}

__host__ __device__
glm::vec3 randomVector(thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u(-1, 1);
	
	glm::vec3 v(u(rng), u(rng), u(rng));

	if (pow(v.length(), 2.f) < 1.f) {
		return v;
	}
	else {
		return randomVector(rng);
	}
}

__host__ __device__
glm::vec3 sampleDisk(float sampleX, float sampleY)
{
	float phi;
	float r;

	float a = 2.f * sampleX - 1.f;
	float b = 2.f * sampleY - 1.f;

	if (a * a > b * b) {
		r = a;
		phi = (PI / 4) * (b / a);
	}
	else {
		r = b;
		//phi = (PI / 4) * (a / b) + (PI / 2);
		phi = (PI / 2) - (PI / 4) * (a / b);
	}

	return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), 0.f);
}

__host__ __device__
float phaseFunction(float g, float cosTheta)
{
	float numerator = (1 - powf(g, 2.f));
	float denominator = (1 + powf(g, 2.f) - powf((2 * g * cosTheta), (3/2)));

	float InvFourPi = 1.f / (4.f * PI);

	return InvFourPi * (numerator / denominator);
}

// http://corysimon.github.io/articles/uniformdistn-on-sphere/
__host__ __device__
glm::vec3 SphereSample(thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float theta = 2.f * PI * u01(rng);
	float phi = acos(1.f - 2.f * u01(rng));
	float x = sin(phi) * cos(theta);
	float y = sin(phi) * sin(theta);
	float z = cos(phi);

	return glm::vec3(x, y, z);
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
	ShadeableIntersection &intersection,
	const Material &m,
	Geom *geoms,
	int num_geom,
	thrust::default_random_engine &rng)
{
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	glm::vec3 wo = -pathSegment.ray.direction;
	glm::vec3 wi(0.f);
	glm::vec3 color(1.f);
	float pdf;

	// Reflective Surface
	if (m.hasReflective) {
		wi = glm::reflect(-wo, intersection.surfaceNormal);

		pathSegment.color *= m.specular.color;
	}
	// Refractive Surface
	else if (m.hasRefractive) {
		float n1 = 1.f;					// air
		float n2 = m.indexOfRefraction;	// material

		// CosTheta > 0 --> ray outside
		// CosTheta < 0 --> ray inside
		bool entering = CosTheta(intersection.surfaceNormal, pathSegment.ray.direction) > 0.f;
		if (!entering) {
			n2 = 1.f / m.indexOfRefraction;
		}

		// Schlick's Approximation
		float r0 = powf((n1 - n2) / (n1 + n2), 2.f);
		float rTheta = r0 + (1 - r0) * powf((1 - glm::abs(glm::dot(intersection.surfaceNormal, pathSegment.ray.direction))), 5.f);

		thrust::uniform_real_distribution<float> u01(0, 1);
		if (rTheta < u01(rng)) {
			wi = glm::normalize(glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, n2));
		}
		else {
			wi = glm::normalize(glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal));
		}

		pathSegment.color *= m.specular.color;
	}
	// Subsurface
	// http://www.davepagurek.com/blog/volumes-subsurface-scattering/
	// https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering
	else if (m.hasSubsurface) {
		// Make a ray at the intersection going in the same direction.
		// This will get updated in the loop.
		Ray nextRay;
		nextRay.origin = intersection.intersectPoint + EPSILON * pathSegment.ray.direction;
		nextRay.direction = pathSegment.ray.direction;

		PathSegment offsetPath = pathSegment;
		offsetPath.ray = nextRay;
		offsetPath.color = glm::vec3(0.f);

		ShadeableIntersection prevIsect = intersection;
		ShadeableIntersection final;

		int	 maxBounce = 5;
		while (maxBounce > 0) {
			// Get the end point of the path 
			// This should still be the same object
			ShadeableIntersection end;
			IntersectRay(offsetPath, geoms, num_geom, end);

			// Should always be the case.
			if (end.t > 0.f) {
				glm::vec3 path = end.intersectPoint - nextRay.origin;

				thrust::uniform_real_distribution<float> u01(0, 1);
				thrust::uniform_real_distribution<float> u(-1, 1);

				// Sample the medium for distance
				float ln = logf(u01(rng));
				float distanceTraveled = -ln / m.density;

				// If the sampled distance is less than the ray then we want to 
				// get a new direction for the next ray and add to the color.
				if (distanceTraveled < path.length()) {

					nextRay.origin = nextRay.origin + glm::normalize(path) * distanceTraveled;
					// Sample the medium for a direction
					nextRay.direction = SphereSample(rng);
					offsetPath.ray = nextRay;

					float transmission = expf(-m.density * distanceTraveled);
					float phase = phaseFunction(m.density, CosTheta(nextRay.direction, end.surfaceNormal));

					// Color gets more muted as we go further along the ray.
					offsetPath.color += m.color * transmission;

					prevIsect = end;
				}
				else {
					final = end;
					break;
				}
			}
			else {
				final = prevIsect;
				break;
			}

			maxBounce--;
		}

		pathSegment.ray = nextRay;
		pathSegment.color = offsetPath.color;

		return;
	}
	// Diffuse Surface
	else {
		wi = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng));

		//pdf = getPdf(wo, wi, normal);
		//color *= INVPI;
		//if (pdf > 0.f) {
		//	pathSegment.color /= pdf;
		//}
		//else {
		//	pathSegment.color = glm::vec3(0.f);
		//}
		// Update color
		//pathSegment.color *= m.color * color;
	}
	
	// Set up ray for the next bounce
	spawnRay(pathSegment, intersection.surfaceNormal, wi, intersection.intersectPoint);
	// Update color 
	pathSegment.color *= m.color * AbsDot(intersection.surfaceNormal, wi);
}



__host__ __device__
glm::vec3 sampleCube(const glm::vec2 &xi, ShadeableIntersection &isect, const Geom &cube)
{
	glm::vec3 p(xi.x - 0.5f, xi.y - 0.5f, 0.f);

	// Check this
	isect.surfaceNormal = glm::normalize(glm::vec3(cube.invTranspose * glm::vec4(0.f, 0.f, 1.f, 1.f)));
	isect.intersectPoint = glm::vec3(cube.transform * glm::vec4(p, 1.f));

	return p;
}

__host__ __device__
bool L(const ShadeableIntersection &isect, const glm::vec3 &w)
{
	if (glm::dot(isect.surfaceNormal, w) > 0.f) {
		return true;
	}

	return false;
}

__host__ __device__
glm::vec3 bsdf_f(
	PathSegment & pathSegment,
	Material &m)
{
	if (m.hasReflective) {
		return glm::vec3(0.f);
	}
	else if (m.hasRefractive) {
		return glm::vec3(0.f);
	}

	// Diffuse
	return m.color * INVPI;
}
