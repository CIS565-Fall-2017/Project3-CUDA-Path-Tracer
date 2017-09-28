#pragma once

#include "stream_compaction/common.h"
#include "intersections.h"
#include "shapefunctions.h"
#include "utilities.h"
#include "utilkern.h"

__host__ __device__ glm::vec3 evaluateFresnelDielectric(float cosThetaI, const float etaI, const float etaT) {
		//561 evaluateFresnel for dieletrics(glass, water, etc)
		float etaI_temp = etaI; float etaT_temp = etaT;
		cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

		//potentially swap indices of refraction
		bool entering = cosThetaI > 0.f;
		if (!entering) {
			float temp = etaI_temp;
			etaI_temp = etaT_temp;
			etaT_temp = temp;
			cosThetaI = std::abs(cosThetaI);
		}

		//compute cos theta I using snells law
		float sinThetaI = std::sqrt(std::max(0.f, 1.f - cosThetaI * cosThetaI));
		float sinThetaT = etaI_temp / etaT_temp * sinThetaI;

		//handle internal reflection
		if (sinThetaT >= 1) {
			return glm::vec3(1.f);
		}

		//equations from 518 in pbrt for calculating fresnel reflectance at the interface of two dielectrics
		float cosThetaT = std::sqrt(std::max(0.f, 1.f - sinThetaT * sinThetaT));
		float r_parallel = ((etaT_temp * cosThetaI) - (etaI_temp * cosThetaT)) /
			((etaT_temp * cosThetaI) + (etaI_temp * cosThetaT));
		float r_perpendicular = ((etaI_temp * cosThetaI) - (etaT_temp * cosThetaT)) /
			((etaI_temp * cosThetaI) + (etaT_temp * cosThetaT));
		float Fr = (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2.f;
		return glm::vec3(Fr);
}

__host__ __device__ bool Refract(const glm::vec3 &wi, const glm::vec3 &n, float eta,
                    glm::vec3& wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = glm::dot(n, wi);
    float sin2ThetaI = std::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = std::sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

//from 561: return this->fresnel->Evaluate(CosTheta(*wi)) * R / AbsCosTheta(*wi);
__host__ __device__
void chooseReflection( PathSegment & path, const ShadeableIntersection& isect, const Material& m,
	float& bxdfPDF, glm::vec3& bxdfColor, thrust::default_random_engine &rng, const bool isDielectric, const float probR) 
{
	//setup
	const glm::vec3 normal = isect.surfaceNormal;
	const glm::vec3 wo = -path.ray.direction;
	const bool entering = glm::dot(normal, wo) > 0.f;
	const float etaI = 1.f;
	const float etaT = m.indexOfRefraction;
	const glm::vec3 colorR = m.color;

	//get bxdfcolor pdf and set path ray
	path.ray.direction = glm::reflect(-wo, normal);//glm assumes incoming ray
	const glm::vec3 wi = path.ray.direction;
	const bool exiting = glm::dot(normal, wi) > 0.f;
	path.ray.origin += (exiting ? normal*EPSILON : -normal*EPSILON);
	if (isDielectric) {//evaluateFresnel will flip for us
		bxdfColor = evaluateFresnelDielectric(cosTheta(normal, wi), etaI, etaT) * colorR / absCosTheta(normal, wi);
		bxdfPDF = probR;
	} else {
		bxdfColor = colorR / absCosTheta(normal, wi);
		bxdfPDF = 1.f;
	}
}

//from 561:
//bool entering = CosTheta(wo) > 0.f;
//float etaI = entering ? etaA : etaB;
//float etaT = entering ? etaB : etaA;
////compute ray direction for specular transmission, return 0 if total internal reflection
//if(!Refract(wo, Faceforward(Normal3f(0,0,1), wo), etaI / etaT, wi)) {
//    return Color3f(0.f);
//}
//*pdf = 1;
//Color3f t = this->T * (1.f - this->fresnel->Evaluate(CosTheta(*wi)));
//return t / AbsCosTheta(*wi);
__host__ __device__
void chooseTransmission( PathSegment & path, const ShadeableIntersection& isect, const Material& m,
	float& bxdfPDF, glm::vec3& bxdfColor, thrust::default_random_engine &rng, const bool isDielectric, const float probR) 
{
	//Determine if incoming or outgoing, adjust accordingly
	const glm::vec3 normal = isect.surfaceNormal;
	const glm::vec3 wo = -path.ray.direction;
	const bool entering = glm::dot (normal, wo) > 0.f;
	const float etaA = 1.f;
	const float etaB = m.indexOfRefraction;
	const float etaI = entering ? etaA : etaB;
	const float etaT = entering ? etaB : etaA;

	//Get wi.
	glm::vec3 wi;
	wi = glm::refract(-wo, Faceforward(normal, wo), etaI / etaT);
	//if (!Refract(wo, Faceforward(normal, wo), etaI / etaT, wi)) {//561
	//	bxdfColor = glm::vec3(0);
	//	bxdfPDF = 0;
	//	path.remainingBounces = -100;
	//	return;
	//}

	//Set Color and PDF and path ray
	const bool exiting = glm::dot(normal, wi) > 0.f;
	path.ray.origin += (exiting ? normal*EPSILON : -normal*EPSILON);
	path.ray.direction = wi;
	const glm::vec3 colorT = m.specular.color;
	if (isDielectric) {//evalfresnel will correct IoR for us
		bxdfColor = colorT * (1.f - evaluateFresnelDielectric(cosTheta(normal, wi), etaA, etaB)) / absCosTheta(normal, wi);
		bxdfPDF = 1.f - probR;
	} else {
		bxdfColor = colorT / absCosTheta(normal, wi);
		bxdfPDF = 1.f;
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
 * - PIck the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRayNaive(
		PathSegment & path,
        const ShadeableIntersection& isect,
        const Material& m,
		float& bxdfPDF,
	    glm::vec3& bxdfColor,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	//1. Determine all bxdfs first and randomly pick one using rng 
	//This prob wont be necessary as everything except glass will just have 1 bxdf

	//2. If we are setting for DL, set the wi base on a randomly sampled point from a random light
	// this takes a diffent path i believe involving light pdfs and such 
	//   If we are not setting for DL, randomly pick a dir for the randomly selected bxdf using rng again(so we don't bias direction based on chosen bxdf)

	//3. If NOT specular take ave pdf of all bxdf's pdfs and ave color of all bxdf's f(wo,wi) (prob don't need this since everything except glass will have 1 bxdf)
	//   If the randomly selected bxdf IS specular then our color is just the color of the specular's f(wo,wi) and the pdf if just 1.f / totalbxdfs i.e. 1/2

	const glm::vec3 normal = isect.surfaceNormal;
	const glm::vec3 wo = -path.ray.direction;
	path.ray.origin = getPointOnRay(path.ray, isect.t);
	float avepdf = 0.f;
	glm::vec3 avecolor(0.f);


	if (!m.hasReflective && !m.hasRefractive) {//just diffuse
		path.ray.origin += normal*EPSILON;
		path.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		const glm::vec3 wi = path.ray.direction;
		bxdfPDF = sameHemisphere(wo, wi,normal) ? cosTheta(normal, wi)*InvPI : 0.f;
		bxdfColor = m.color * InvPI;
	} else if ( m.hasReflective && !m.hasRefractive) {//just reflective
		chooseReflection(path, isect, m, bxdfPDF, bxdfColor, rng, false, 1.f);

	} else if (!m.hasReflective &&  m.hasRefractive) {//just transmissive
		chooseTransmission(path, isect, m, bxdfPDF, bxdfColor, rng, false, 0.f);
		
	} else if (m.hasReflective &&  m.hasRefractive) {//relective and transmissive
		//determine reflect probability based on luminance 
		const glm::vec3 colorR = m.color;
		const glm::vec3 colorT = m.specular.color;
		const float colorRLum = colorR.r*0.2126f + colorR.g*0.7152f + colorR.b*0.0722f;
		const float colorTLum = colorT.r*0.2126f + colorT.g*0.7152f + colorT.b*0.0722f;
		const float probR = colorRLum / (colorRLum + colorTLum);

		thrust::uniform_real_distribution<float> u01(0, 1);
		if (u01(rng) < probR) {
			chooseReflection(path, isect, m, bxdfPDF, bxdfColor, rng, true, probR);

		} else {
			chooseTransmission(path, isect, m, bxdfPDF, bxdfColor, rng, true, probR);
		}
	}
	path.ray.direction = glm::normalize(path.ray.direction);
}

//MIS light sampling
__host__ __device__
glm::vec3 sampleLight( const Geom& randlight, const Material& mlight,
	const glm::vec3& pisect, const glm::vec3& nisect,
	thrust::default_random_engine &rng,
	glm::vec3& widirect, float& pdfdirect) 
{

	glm::vec3 nlightsamp; 
	glm::vec3 plightsamp = surfaceSampleShape(nlightsamp,
			randlight, pisect, nisect, rng, pdfdirect);

	if ( 0.f >= pdfdirect || glm::all(glm::equal(pisect, plightsamp)) ) {
		pdfdirect = 0;
		return glm::vec3(0.f);
	}
	widirect = glm::normalize(plightsamp - pisect);
	return (glm::dot(nlightsamp, -widirect)) > 0.f ? 
		mlight.color*mlight.emittance : glm::vec3(0.f);
}

__host__ __device__
void bxdf_FandPDF(const glm::vec3& wo, const glm::vec3& wi, 
	const glm::vec3& normal, const Material& m, 
	glm::vec3& bxdfColor, float& bxdfPDF) 
{
	if (!m.hasReflective && !m.hasRefractive) {//pure diffuse
		const float wodotwi = glm::dot(wo, wi);
		bxdfPDF = sameHemisphere(wo, wi, normal) ? cosTheta(normal, wi)*InvPI : 0.f;
		bxdfColor = m.color * InvPI;
	} else if (m.hasReflective || m.hasRefractive) {
		bxdfPDF = 0;
		bxdfColor = glm::vec3(0);
	}
	//TODO: other non specular bxdfs?
}

__host__ __device__
float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    const float f = nf * fPdf, g = ng * gPdf;
    const float denom = f*f + g*g;
    if(denom < FLT_EPSILON) {
        return 0.f;
    }
    return (f*f) / (denom);
}

__host__ __device__
float lightPdfLi(const Geom& randlight, const glm::vec3& pisect, 
	const glm::vec3& wi) 
{
	glm::vec3 pisect_thislight; glm::vec3 nisect_thislight;
	Ray wiWray; wiWray.origin = pisect; wiWray.direction = wi;


	bool outside;
    if(0.f > shapeIntersectionTest(randlight, wiWray, pisect_thislight, nisect_thislight,outside)) {
        return 0.f;
    }

    const float coslight = glm::dot(nisect_thislight, -wi);
	//uncomment if twosided
	//coslight = glm::abs(coslight);

	const float denom = coslight * getAreaShape(randlight);
    if(denom > 0.f) {
		return glm::distance2(pisect, pisect_thislight) / denom;
    } else {
        return 0.f;
    }

}

__host__ __device__
glm::vec3 Le(const Material& misect, const glm::vec3 nisect, 
	const glm::vec3 source_dir)
{
	if(0.f < misect.emittance && 0.f < glm::dot(source_dir, nisect)) {
		return misect.color*misect.emittance;
	} 
	return glm::vec3(0);
}
