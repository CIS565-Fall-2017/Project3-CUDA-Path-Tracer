#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
#define pushoutTolerance 0.01f;
namespace Global {
	typedef thrust::tuple<PathSegment, ShadeableIntersection> Tuple;
	class MaterialCmp
	{
	public:
		__host__ __device__ bool operator()(const Tuple &a, const Tuple &b)
		{
			return a.get<1>().materialId < b.get<1>().materialId;
		}
	};
	template <typename T>
	__host__ __device__ void swap(T &a, T &b)
	{
		T tmp(a);
		a = b;
		b = tmp;
	}
	__host__ __device__ float cosWeightedHemispherePdf(
		const glm::vec3 &normal, const glm::vec3 &dir)
	{
		return fmax(0.f, glm::dot(normal, dir)) / PI;
	}

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
		glm::vec3 dir = glm::normalize(up * normal
			+ cos(around) * over * perpendicularDirection1
			+ sin(around) * over * perpendicularDirection2);
		return dir;
	}
}
namespace LightSample {
	//Here we use cube to simulate a plan. We suppose the height of the cube can be ignore.
	__host__ __device__ float CubeSamplePdf(const Geom& cube,const glm::vec3&origin,const glm::vec3& Dir) {
		Ray r{ origin, Dir };
		glm::vec3 intersect;
		glm::vec3 normal, tan, bit;
		bool outside = true;
		float t = boxIntersectionTest(cube, r, intersect, normal, tan, bit, outside);
		glm::vec3 localNormal = multiplyMV(cube.inverseTransform, glm::vec4(normal, 0));
		if (t == -1 || glm::length2(localNormal - glm::vec3(0, -1, 0)) > FLT_EPSILON) return 0;
		float R_L = glm::length2(intersect - origin);
		float pdf= 1 / (1 * cube.scale[0] * cube.scale[2]);
		return R_L / (fabs(glm::dot(Dir, normal)) * 1 * cube.scale[0] * cube.scale[2]);
	}
	__host__ __device__ void PlaneSample_Li(const Geom& cube,
		const glm::vec3 origin,
		glm::vec3& woW,
		int numGeo,
		const Geom* geoms,
		float& pdf,
		bool& occlusion,
		thrust::default_random_engine &rng,
		thrust::uniform_real_distribution<float> &u01) {

		float t_min;
		glm::vec3 isect_normal;
		glm::vec3 tan;
		glm::vec3 bit;
		int hit_geom_index = -1;
		
		float xi[2] = { u01(rng),u01(rng) };
		glm::vec3 intersect(xi[0], -0.5f, xi[1]);
		intersect = multiplyMV(cube.transform, glm::vec4(intersect, 1));
		woW = intersect - origin;
		float R_L = glm::length2(woW);
		woW = glm::normalize(woW);
		Ray r{ origin,woW };
		SearchIntersection::BruteForceSearch(t_min, hit_geom_index, isect_normal, tan, bit, r, geoms, numGeo);	
		pdf = R_L / (fabs(glm::dot(woW, isect_normal) * 1 * cube.scale[0] * cube.scale[2]));
		occlusion = true;
		if (t_min > 0.f && hit_geom_index == cube.idx)
			occlusion = false;
	}
	
}
namespace Fresnel {
	__host__ __device__ float Dielectric(float cosThetaI, float etaI, float etaT)
	{
		//Computing cosThetaT using Snell's Law
		float sinThetaI = sqrt(fmax(0.0f, 1 - cosThetaI*cosThetaI));
		float sinThetaT = etaI / etaT*sinThetaI;
		float cosThetaT = sqrt(fmax(0.0f, 1 - sinThetaT*sinThetaT));

		//    //Handle internal reflection
		if (sinThetaT>1.0f)
			return 1.f;
		float r_parl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
			((etaT * cosThetaI) + (etaI * cosThetaT)+FLT_EPSILON);
		float r_perp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
			((etaI * cosThetaI) + (etaT * cosThetaT)+FLT_EPSILON);
		float result = (r_parl*r_parl + r_perp*r_perp) / 2;
		return result;
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
namespace BSDF {
	__host__ __device__ float LambertPDF(const glm::vec3 &normal, const glm::vec3 &dir) {
		return Global::cosWeightedHemispherePdf(normal, dir);
	}
	__host__ __device__ glm::vec3 LambertSample_f(const glm::vec3& normal, glm::vec3& woW, 
		float& pdf, thrust::default_random_engine &rng,const glm::vec3& mc) {
		woW = Global::calculateRandomDirectionInHemisphere(normal, rng);
		pdf = LambertPDF(normal, woW);
		return mc / PI;
	}
	__host__ __device__ float SpecularPDF() {
		return 0.0f;
	}
	__host__ __device__ glm::vec3 SpecularSample_f(const glm::vec3& wiW,const glm::vec3& normal, glm::vec3& woW,
		float& pdf, thrust::default_random_engine &rng, const glm::vec3& mc) {
		woW = glm::normalize(glm::reflect(wiW, normal));
		pdf = 1.0;
		float absCosTheta = fabs(glm::dot(normal, woW));
		return mc/ absCosTheta;
	}

	__host__ __device__ float GlassPDF() {
		return 0.0f;
	}

	__host__ __device__ glm::vec3 GlassBTDF(const glm::vec3& wiW, glm::vec3 normal, glm::vec3& woW,
		float& pdf, thrust::default_random_engine &rng, const glm::vec3& mc,float etat) {
		thrust::uniform_real_distribution<float> u01(0, 1);
		float cosThetaI=glm::dot(wiW, normal);
		float etai=1.0f;
		float fresnel;
		glm::vec3 REFL_Dir, REFR_Dir;
		if (cosThetaI > 0.f)
		{
			normal = -normal;
			Global::swap(etai,etat);
		}

		REFL_Dir = glm::normalize(glm::reflect(wiW, normal));
		REFR_Dir = glm::normalize(glm::refract(wiW, normal, etai / etat));
		fresnel = Fresnel::Dielectric(fabs(cosThetaI), etai, etat);

		float rn = u01(rng);
		if (rn < fresnel || glm::length2(REFR_Dir) < FLT_EPSILON){ // deal with total internal reflection
			woW = REFL_Dir;
		}
		else {
			woW = REFR_Dir;
		}
			
		pdf = 1;
		return mc / fabs(cosThetaI);
	}
}




__host__ __device__
void scatterRay(
	PathSegment & pathSegment,
	const Geom * geoms,
	const Geom** dev_lights,
	glm::vec3 intersect,
	glm::vec3 normal,
	glm::mat3 WorldToTangent,
	glm::mat3 TangentToWorld,
    const Material &m,
    thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	//lambert
	float pdf = 0;
	glm::vec3 f;
	glm::vec3 wiW = pathSegment.ray.direction;
	glm::vec3 woW;
	glm::vec3 wi = WorldToTangent*wiW;

	if (m.hasReflective) {
		//pathSegment.ray.direction= glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
		f = BSDF::SpecularSample_f(wiW, normal, woW, pdf, rng, m.color);

	}
	else if (m.hasRefractive) {
		f = BSDF::GlassBTDF(wiW, normal, woW,pdf, rng, m.color, m.indexOfRefraction);
	}
	else {
		//Lambert Case
		f = BSDF::LambertSample_f(normal, woW, pdf, rng, m.color);

	}
	pathSegment.ray.direction = woW;
	pathSegment.ray.origin = intersect + pathSegment.ray.direction*pushoutTolerance;
	//pathSegment.color *= m.color;
	pathSegment.color *= f * fabs(glm::dot(normal, pathSegment.ray.direction)) / (pdf + FLT_EPSILON);
	

}
