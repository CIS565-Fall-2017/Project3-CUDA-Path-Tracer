#pragma once

#include "intersections.h"

__host__ __device__ bool isBlack(const glm::vec3 c) {
    return (c.r <= 0.f && c.g <= 0.f && c.b <= 0.f);
}

__host__ __device__ float absDot(const glm::vec3 a, const glm::vec3 b) {
    return glm::abs(glm::dot(a, b));
}

__host__ __device__ float cosTheta(const glm::vec3 a, const glm::vec3 b) {
	return glm::dot(a, b);
}

__host__ __device__ float absCosTheta(const glm::vec3 a, const glm::vec3 b) {
    return glm::abs(glm::dot(a, b));
}

__host__ __device__ bool sameHemisphere(const glm::vec3 a, const glm::vec3 b) {
	return glm::dot(a, b) > 0.f;
}
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

__host__ __device__ glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
    return (glm::dot(n, v) < 0.f) ? -n : n;
}

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
void chooseTransmission(
	PathSegment & path,
	const ShadeableIntersection& isect,
	const Material& m,
	float& bxdfPDF,
	glm::vec3& bxdfColor,
	thrust::default_random_engine &rng, const bool isDielectric, const float probR) 
{
	//Determine if incoming or outgoing, adjust accordingly
	const glm::vec3 normal = isect.surfaceNormal;
	const glm::vec3 wo = -path.ray.direction;
	const bool entering = cosTheta(normal, wo) > 0.f;
	const glm::vec3 norm = (entering ? normal : -normal);
	const float etaA = 1.f;
	const float etaB = m.indexOfRefraction;
	const float etaI = entering ? etaA : etaB;
	const float etaT = entering ? etaB : etaA;

	//Get wi.
	glm::vec3 wi;
	//wi = glm::refract(wo, norm, etaI / etaT);//refract expects ray pointing to where it came from?
	//glm::refract
	const float eta = etaI / etaT;
    float k = 1.f - eta * eta * (1.f - glm::dot(norm, wo) * glm::dot(norm, wo));
	if (k < 0.f) {
		wi = glm::vec3(0.f);
	} else {
		wi = eta * wo - (eta * glm::dot(norm, wo) + sqrtf(k)) * norm;
	}

	//if(!Refract(-wo, Faceforward(normal, wo), etaI / etaT, wi)) {//Why is this -wo change neccessary(not in 561 code above)
	//	bxdfColor = glm::vec3(0.f);//total internal reflection
	//	bxdfPDF = 1.f;
	//	return;
	//}

	//Set Color and PDF and path ray
	const bool exiting = cosTheta(normal, wi) > 0.f;
	path.ray.origin += (exiting ? normal*EPSILON : -normal*EPSILON);
	path.ray.direction = wi;
	const glm::vec3 colorT = m.specular.color;
	if (isDielectric) {
		bxdfColor = colorT * (1.f - evaluateFresnelDielectric(cosTheta(norm, wi), etaI, etaT)) / absCosTheta(norm, wi);
		bxdfPDF = 1.f - probR;
	} else {
		bxdfColor = colorT / absCosTheta(norm, wi);
		bxdfPDF = 1.f;
	}
}

__host__ __device__
void chooseReflection(
	PathSegment & path,
	const ShadeableIntersection& isect,
	const Material& m,
	float& bxdfPDF,
	glm::vec3& bxdfColor,
	thrust::default_random_engine &rng, const bool isDielectric, const float probR) 
{
	const glm::vec3 normal = isect.surfaceNormal;
	const glm::vec3 wo = -path.ray.direction;
	const bool entering = cosTheta(normal, wo) > 0.f;
	const glm::vec3 norm = (entering ? normal : -normal);
	const float etaA = 1.f;
	const float etaB = m.indexOfRefraction;
	const float etaI = entering ? etaA : etaB;
	const float etaT = entering ? etaB : etaA;
	const glm::vec3 colorR = m.color;

	path.ray.origin += norm*EPSILON;
	path.ray.direction = glm::reflect(-wo, norm);//glm assumes incoming ray
	const glm::vec3 wi = path.ray.direction;
	//from 561: return this->fresnel->Evaluate(CosTheta(*wi)) * R / AbsCosTheta(*wi);
	if (isDielectric) {
		bxdfColor = evaluateFresnelDielectric(cosTheta(norm, wi), etaI, etaT) * colorR / absCosTheta(norm, wi);
		bxdfPDF = probR;
	} else {
		bxdfColor = colorR / absCosTheta(norm, wi);
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
 * - Pick the split based on the intensity of each material color, and divide
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
		bxdfPDF = sameHemisphere(wo, wi) ? cosTheta(normal, wi)*InvPi: 0.f;
		bxdfColor = m.color * InvPi;

	} else if ( m.hasReflective && !m.hasRefractive) {//just reflective
		chooseReflection(path, isect, m, bxdfPDF, bxdfColor, rng, false, 1.f);

	} else if (!m.hasReflective &&  m.hasRefractive) {//just refraction
		chooseTransmission(path, isect, m, bxdfPDF, bxdfColor, rng, true, 0);
		
	} else if (m.hasReflective &&  m.hasRefractive) {//relective and refractive
		//determine reflect probability based on luminance 
		const glm::vec3 colorR = m.color;
		const glm::vec3 colorT = m.specular.color;
		const float colorRLum = colorR.r*0.2126f + colorR.g*0.7152f + colorR.b*0.0722f;
		const float colorTLum = colorT.r*0.2126f + colorT.g*0.7152f + colorT.b*0.0722f;
		const float probR = colorRLum / (colorRLum + colorTLum);

		thrust::uniform_real_distribution<float> u01(0, 1);
		if (u01(rng) < probR) {//reflect
			chooseReflection(path, isect, m, bxdfPDF, bxdfColor, rng, true, probR);
		} else {//refract
			chooseTransmission(path, isect, m, bxdfPDF, bxdfColor, rng, true, probR);
		}
	}
	path.ray.direction = glm::normalize(path.ray.direction);
}
