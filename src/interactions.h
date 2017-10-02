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
			float& pdf,
			glm::vec3 normal, thrust::default_random_engine &rng,
			thrust::uniform_real_distribution<float> &u01) {

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
		pdf = cosWeightedHemispherePdf(normal, dir);
		return dir;
	}
	__host__ __device__ __forceinline__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = nf * fPdf;
		float g = ng * gPdf;
		return f * f / (f * f + g * g);
	}
}
namespace shapeSample {
	__host__ __device__ __forceinline__ float uniformConePdf(float costheta_max)
	{
		return 1.0f / (TWO_PI * (1.0f - costheta_max));
	}

	__host__ __device__ glm::vec3 uniformSampleCone(
		float cos_theta_max,
		thrust::default_random_engine &rng,
		thrust::uniform_real_distribution<float> &u01)
	{
		float u = u01(rng);
		float v = u01(rng);

		float costheta = (1.0f - u) + u * cos_theta_max;
		float sintheta = sqrtf(1.0f - costheta * costheta);
		float phi = v * TWO_PI;

		return glm::vec3(cosf(phi) * sintheta, sinf(phi) * sintheta, costheta);
	}

	__host__ __device__ void getCone2LocalT(const glm::vec3 &cz, glm::mat3 &cone2local)
	{
		glm::vec3 cx, cy;

		cx = glm::cross(glm::vec3(0, 1, 0), cz);

		if (glm::length2(cx) < FLT_EPSILON)
		{
			cx = glm::vec3(1, 0, 0);
			cy = glm::vec3(0, 0, -1);
		}
		else
		{
			cx = glm::normalize(cx);
			cy = glm::normalize(glm::cross(cz, cx));
		}

		cone2local[0] = cx;
		cone2local[1] = cy;
		cone2local[2] = cz;
	}

	__host__ __device__ void concentricSamplingDisk(float u, float v, float &x_ret, float &y_ret)
	{
		float r, phi;
		float x = 2 * u - 1;    // [0, 1] -> [-1, 1]
		float y = 2 * v - 1;

		// handle degeneracy at the origin
		if (fabsf(x) < FLT_EPSILON && fabsf(y) > FLT_EPSILON)
		{
			x_ret = 0.f;
			y_ret = 0.f;
		}

		if (x > -y)
		{
			if (x > y)
			{
				// first region
				r = x;
				phi = y / x;
			}
			else
			{
				r = y;
				phi = 2 - x / y;
			}
		}
		else
		{
			if (x < y)
			{
				r = -x;
				phi = 4 + y / x;
			}
			else
			{
				r = -y;
				phi = 6 - x / y;
			}
		}

		phi *= 0.25f * PI;
		x_ret = r * cosf(phi);
		y_ret = r * sinf(phi);
	}
}
namespace LightSample {
	using namespace shapeSample;
	//Here we use cube to simulate a plan. We suppose the height of the cube can be ignore.
	__host__ __device__ float PlaneSamplePdf(const Geom& cube, const glm::vec3&origin, const glm::vec3& Dir) {
		Ray r{ origin, Dir };
		glm::vec3 intersect;
		glm::vec3 normal;
		bool outside = true;
		float t = boxIntersectionTest(cube, r, intersect, normal, outside);
		glm::vec3 localNormal = multiplyMV(cube.inverseTransform, glm::vec4(normal, 0));
		if (t == -1 || glm::length2(localNormal - glm::vec3(0, -1, 0)) > FLT_EPSILON) return 0;
		float R_L = glm::length2(intersect - origin);
		return R_L / (fabs(glm::dot(Dir, normal)) * 1 * cube.scale[0] * cube.scale[2]);
	}
	__host__ __device__ void PlaneSample(const Geom& cube,
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
		int hit_geom_index = -1;

		float xi[2] = { u01(rng),u01(rng) };
		glm::vec3 intersect(xi[0], -0.5f, xi[1]);
		intersect = multiplyMV(cube.transform, glm::vec4(intersect, 1));
		woW = intersect - origin;
		float R_L = glm::length2(woW);
		woW = glm::normalize(woW);
		Ray r;
		r.origin = origin;
		r.direction = woW;

		SearchIntersection::BruteForceSearch(t_min, hit_geom_index, isect_normal, r, geoms, numGeo);
		pdf = R_L / (fabs(glm::dot(woW, isect_normal) * cube.scale[0] * cube.scale[2]));
		occlusion = true;
		if (t_min > 0.f && hit_geom_index == cube.idx)
			occlusion = false;
	}

	//Jian's code for sampling a sphere light
	__host__ __device__ float sphereLightPdf(const Geom &sphere, const glm::vec3 &isecPt, const glm::vec3 &dir)
	{
		glm::vec3 p = glm::vec3(sphere.inverseTransform * glm::vec4(isecPt, 1.0f));
		glm::vec3 d = glm::normalize(glm::vec3(sphere.inverseTransform * glm::vec4(dir, 0.f)));
		float sin_theta_max2 = 0.25f / glm::distance2(glm::vec3(0.0f), p);
		float cos_theta_max = sqrtf(fmax(0.0f, 1.0f - sin_theta_max2));
		float cos_theta = glm::dot(glm::normalize(-p), d);
		return cos_theta >= cos_theta_max ? uniformConePdf(cos_theta_max) : 0.f;
	}

	__host__ __device__ void sampleLight_Sphere(
		glm::vec3 &sample_dir,
		bool &isBlocked,
		float &pdf,
		const Geom &sphere,
		const glm::vec3 &isecPt,
		const Geom *geoms,
		int numGeoms,
		thrust::default_random_engine &rng,
		thrust::uniform_real_distribution<float> &u01)
	{
		glm::vec3 p = glm::vec3(sphere.inverseTransform * glm::vec4(isecPt, 1.0f));   // local p
		float sin_theta_max2 = 0.25f / glm::distance2(glm::vec3(0.0f), p);
		float cos_theta_max = sqrtf(fmax(0.0f, 1.0f - sin_theta_max2));

		// get the transform from cone local space to sphere local space to
		// make it easier to sample the cone
		glm::vec3 cz = glm::normalize(-p);
		glm::mat3 cone2local;
		glm::vec3 cone_sample_dir = uniformSampleCone(cos_theta_max, rng, u01);

		getCone2LocalT(cz, cone2local);
		sample_dir = glm::normalize(glm::vec3(sphere.transform * glm::vec4(cone2local * cone_sample_dir, 0.0f))); // to world space

		Ray r = { isecPt, sample_dir };
		float t;
		int hitGeomIdx;
		glm::vec3 normal;
		SearchIntersection::BruteForceSearch(t, hitGeomIdx, normal, r, geoms, numGeoms);

		pdf = uniformConePdf(cos_theta_max);
		isBlocked = true;
		if (t > 0.f && hitGeomIdx == sphere.idx)
		{
			isBlocked = false;
		}
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
	__host__ __device__ glm::vec3 Lambert_f(const glm::vec3& mc) {
		return mc / PI;
	}
	__host__ __device__ glm::vec3 LambertSample_f(const glm::vec3& normal, glm::vec3& woW,
		float& pdf, thrust::default_random_engine &rng,
		thrust::uniform_real_distribution<float> &u01, const glm::vec3& mc) {
		woW = Global::calculateRandomDirectionInHemisphere(pdf,normal, rng,u01);
		return mc / PI;
	}
	__host__ __device__ float SpecularPDF() {
		return 0.0f;
	}
	__host__ __device__ glm::vec3 Specular_f() {
		return glm::vec3(0.f);
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
	__host__ __device__ glm::vec3 Glass_f() {
		return glm::vec3(0.f);
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
	int numgeo,
	const Geom** dev_lights,
	int lightnum,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material &m,
	const Material* mat,
	thrust::default_random_engine &rng,
	thrust::uniform_real_distribution<float> &u01) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	//lambert
	float pdf = 0;
	glm::vec3 f;
	glm::vec3 wiW = pathSegment.ray.direction;
	glm::vec3 woW;

	if (m.hasReflective) {
		//pathSegment.ray.direction= glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
		f = BSDF::SpecularSample_f(wiW, normal, woW, pdf, rng, m.color);

	}
	else if (m.hasRefractive) {
		f = BSDF::GlassBTDF(wiW, normal, woW,pdf, rng, m.color, m.indexOfRefraction);
	}
	else {
		//Lambert Case
		f = BSDF::LambertSample_f(normal, woW, pdf, rng, u01, m.color);

		glm::vec3 LightDir;
		glm::vec3 ScatterDir;
		glm::vec3 f_scattering;
		glm::vec3 f_light;
		//intersect = intersect + normal*pushoutTolerance;

		int lightidx = static_cast<int>(u01(rng) * lightnum);
		lightidx = (lightidx <= lightnum - 1) ? lightidx : (lightnum - 1);
		const Geom &lightSrc = *dev_lights[lightidx];
		const Material &lightMat = mat[lightSrc.materialid];
		glm::vec3 lightColor = lightMat.color * lightMat.emittance;

		float lightPdf_WL, scatterPdf_WL, lightPdf_WS, scatterPdf_WS;
		bool occlusion = false;
		float WL, WS;


		LightSample::sampleLight_Sphere(LightDir, occlusion, lightPdf_WL, lightSrc, intersect + 0.001f * normal, geoms, numgeo, rng, u01);
		scatterPdf_WL = BSDF::LambertPDF(normal, LightDir);
		f_light = BSDF::Lambert_f(m.color);

		f_scattering = BSDF::LambertSample_f(normal, ScatterDir, scatterPdf_WS, rng, u01, m.color);
		lightPdf_WS = LightSample::sphereLightPdf(lightSrc, intersect + 0.001f * normal, ScatterDir);

		WL = Global::powerHeuristic(1, lightPdf_WL, 1, scatterPdf_WL);
		WS = Global::powerHeuristic(1, scatterPdf_WS, 1, lightPdf_WS);

		if (!occlusion) {
			if (pathSegment.remainingBounces == 1) {
				pathSegment.color += pathSegment.beta_loop*f_light * lightColor;
			}
			else {
				glm::vec3 DirColor = f_light * lightColor * fabs(glm::dot(normal, LightDir)) *
					WL / (lightPdf_WL + FLT_EPSILON);
				glm::vec3 ScatterColor = lightPdf_WS*scatterPdf_WS ? f_scattering * lightColor * fabs(glm::dot(normal, ScatterDir)) *
					WS / (scatterPdf_WS + FLT_EPSILON) : glm::vec3(0.f);
				pathSegment.color +=
					(pathSegment.beta_loop *(DirColor + ScatterColor));
			}

		}

	}
	pathSegment.ray.direction = woW;
	pathSegment.ray.origin = intersect + pathSegment.ray.direction*pushoutTolerance;
	//pathSegment.color *= m.color;
	//Naive
	//pathSegment.color *= f * fabs(glm::dot(normal, pathSegment.ray.direction)) / (pdf + FLT_EPSILON);
	//MIS
	pathSegment.beta_loop*= f * fabs(glm::dot(normal, pathSegment.ray.direction)) / (pdf + FLT_EPSILON);

}
