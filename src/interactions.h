#pragma once

#include "stream_compaction/common.h"
#include "intersections.h"
#include "shapefunctions.h"
#include "utilities.h"
#include "utilkern.h"
#include "glm/gtx/component_wise.hpp"

__host__ __device__ glm::vec3 evaluateFresnelDielectric(float cosThetaI, const float etaI, const float etaT) {
		//561 evaluateFresnel for dieletrics(glass, water, etc)
		float etaI_temp = etaI; float etaT_temp = etaT;
		cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

		//potentially swap indices of refraction
		const bool entering = cosThetaI > 0.f;
		if (!entering) {
			float temp = etaI_temp;
			etaI_temp = etaT_temp;
			etaT_temp = temp;
			cosThetaI = glm::abs(cosThetaI);
		}

		//compute cos theta I using snells law

		const float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
		const float sinThetaT = etaI_temp / etaT_temp * sinThetaI;

		//handle internal reflection
		if (sinThetaT >= 1) {
			return glm::vec3(1.f);
		}

		//equations from 518 in pbrt for calculating fresnel reflectance at the interface of two dielectrics
		const float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));
		const float r_parallel = ((etaT_temp * cosThetaI) - (etaI_temp * cosThetaT)) /
		 	((etaT_temp * cosThetaI) + (etaI_temp * cosThetaT));
		const float r_perpendicular = ((etaI_temp * cosThetaI) - (etaT_temp * cosThetaT)) /
		 	((etaI_temp * cosThetaI) + (etaT_temp * cosThetaT));
		const float Fr = (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2.f;
		return glm::vec3(Fr);
}

__host__ __device__ bool Refract(const glm::vec3 &wi, const glm::vec3 &n, float eta,
                    glm::vec3& wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = glm::dot(n, wi);
    float sin2ThetaI =glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = glm::sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

//from 561: return this->fresnel->Evaluate(CosTheta(*wi)) * R / AbsCosTheta(*wi);
__host__ __device__
void chooseReflection( PathSegment & path, const ShadeableIntersection& isect, const Material& m,
	float& bxdfPDF, glm::vec3& bxdfColor, thrust::default_random_engine &rng, const bool isDielectric, const float probR) 
{
	//setup
	path.ray.origin = getPointOnRay(path.ray, isect.t);
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
	path.ray.origin = getPointOnRay(path.ray, isect.t);
	const glm::vec3 normal = isect.surfaceNormal;
	const glm::vec3 wo = -path.ray.direction;
	const bool entering = glm::dot (normal, wo) > 0.f;
	const float etaA = 1.f;
	const float etaB = m.indexOfRefraction;
	const float etaI = entering ? etaA : etaB;
	const float etaT = entering ? etaB : etaA;

	//Get wi.
	glm::vec3 wi;
	//wi = glm::refract(-wo, Faceforward(normal, wo), etaI / etaT);//same as below
	if (!Refract(wo, Faceforward(normal, wo), etaI / etaT, wi)) {//561
		//when total internal reflection occurs all light is reflected
		//so we can just reflect the ray and ignore doing fresnel if it's dielectric
		//sure, it's likely to get russian rouletted inside a perfect sphere 
		//but for other shapes we should reflect and keep going
		//otherwise it will look dimmer than it should
		chooseReflection(path,isect,m,bxdfPDF,bxdfColor,rng,false,1);
		return;
	}

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


//MIS light sampling
__host__ __device__
glm::vec3 sampleLight( const Geom& randlight, const Material& mlight,
	const glm::vec3& pisect, const glm::vec3& nisect,
	thrust::default_random_engine &rng,
	glm::vec3& widirect, float& pdfdirect, glm::vec3& plightsamp, glm::vec3& nlightsamp) 
{

	//glm::vec3 nlightsamp; 
	plightsamp = surfaceSampleShape(nlightsamp,
			randlight, pisect, nisect, rng, pdfdirect);

	if ( 0.f >= pdfdirect || glm::all(glm::equal(pisect, plightsamp)) ) {
		pdfdirect = 0;
		return glm::vec3(0.f);
	}
	widirect = glm::normalize(plightsamp - pisect);
	return (glm::dot(nlightsamp, -widirect)) > -0.05 ?  
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
float powerHeuristic(const int nf, const float fPdf, const int ng, const float gPdf) {
    const float f = nf * fPdf, g = ng * gPdf;
    const float denom = f*f + g*g;
    if(denom < FLT_EPSILON) {
        return 0.f;
    }
    return (f*f) / denom;
}

__host__ __device__
float powerHeuristic3(const int nf, const float fPdf, 
	const int ng, const float gPdf, const int nk, const float kPdf) {
	const float f = nf * fPdf, g = ng * gPdf, k = nk * kPdf;
	const float denom = f*f + g*g + k*k;
    if(denom < FLT_EPSILON) {
        return 0.f;
    }
    return (f*f) / denom;
}

__host__ __device__
float lightPdfLi(const Geom& randlight, const glm::vec3& pisect, 
	const glm::vec3& wi, const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices
	) 
{
	glm::vec3 pisect_thislight; glm::vec3 nisect_thislight;
	Ray wiWray; wiWray.origin = pisect; wiWray.direction = wi;

	bool outside;
    if(0.f > shapeIntersectionTest(randlight, wiWray, pisect_thislight, nisect_thislight,outside, dev_BVHNodes, dev_TriIndices, dev_TriVertices)) {
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
	//Assumes lights are not two sided
	if(0.f < misect.emittance && 0.f < glm::dot(source_dir, nisect)) {
		return misect.color*misect.emittance;
	} 
	return glm::vec3(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////                                    For BSSRDF Read:					//////////////////////////
//////Jensen01 dipole model bssrdf "a practical model for subsurface light transport"///////////////////
///////see also King13 "bssrdf importance sampling" by solid angle and sony imageworks///////////////////////
///////see also ch5 of "Towards realstic image synthesis of scattering Materials" by Craig Donner////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//TODO: cite pages in papers, find better names for things so they 
//connect to the papers better

//jensen page 5 top right
//marble: sigA(units are per mm) = 0.0021, 0.0041, 0.0071 
//marble: sigSPrime = 2.19, 2.62, 3
//marble: eta = 1.5
//marble: diffuse reflectance should be = 0.83, 0.79, 0.75 

//NOTE: sigTr below is actually just luminance of sigTr

//First some pdf and sampling stuff from "bssdrf importance sampling" from solid angle and imageworks

__host__ __device__
glm::vec2 getDiscXY(thrust::default_random_engine& rng, const float sigTr, const float rMax2) {
	//NOTE: should also just try the inverted jensen01 equation instead of king13
	// 1/2v in the quantized diffusion paper is sigTr
	thrust::uniform_real_distribution<float> u01(0, 1);
	const float neg2v = -1.f / sigTr;
	const float r = glm::sqrt(neg2v*glm::log(1.f - u01(rng)*(1.f - glm::exp(-rMax2*sigTr)) ));
	const float theta = TWO_PI * u01(rng);
	return glm::vec2(r*glm::cos(theta), r*glm::sin(theta));
}
__host__ __device__ 
float getSingleScatterDist(thrust::default_random_engine& rng, const float sigT) {
	//jensen page 6, sPrime0, importance sampled single scattering distance
	thrust::uniform_real_distribution<float> u01(0, 1);
	return -glm::log(u01(rng)) / sigT;
}


__host__ __device__
float getSingleScatterPdf(const float r, const float sigT) {
	//jensen page 6
	return sigT*glm::exp(-sigT*r);
	//return glm::exp(-sigTr*r);
}


__host__ __device__
float getSinglePlanarGaussianRd(const float r, const float sigTr) {
	//king13
	return InvPI*sigTr*glm::exp(-sigTr*r*r);
}

__host__ __device__
float getSinglePlanarGaussianRd_Reproject(const glm::vec3& P,
	const glm::vec3 PHit, const glm::vec3& normal, const float sigTr) 
{
	//king13
	const glm::vec3 d = PHit - P;//want to move the head of this vector towards the sampling plane
	const glm::vec3 projectedToChosenTangentPlane =
		d - normal*glm::dot(d, normal);//when you dot with an axis, you strip out that component from the vector, so lets remove this portion from teh original so it lies in the plane
	//orogonal to the axis with which we dotted
	//could also dot with the other axes and get the components of the vector we want to build
	return InvPI*sigTr*glm::exp(-sigTr*
	glm::length2(projectedToChosenTangentPlane));
}

__host__ __device__ 
float getPdfImportanceSampledDisk(const float r, const float sigTr, const float rMax2) {
	const float Rdr = getSinglePlanarGaussianRd(r, sigTr);
	return Rdr / (1.f - glm::exp(-rMax2*sigTr));
}

__host__ __device__
float getPdfImportanceSampledDisk_Reproject(const glm::vec3&P, 
	const glm::vec3 PHit, const glm::vec3& normal, const float sigTr, const float rMax2)
{
	const float Rdr_reproject = getSinglePlanarGaussianRd_Reproject(P, PHit, normal, sigTr);
	return Rdr_reproject / (1.f - glm::exp(-rMax2*sigTr));
}

//Not sure what this is doing in the paper, could try after the reprojecting stuff.
//doesn't really line up with the MIS stuff
__host__ __device__
float getIncomingIrradianceContributionWeight(const glm::vec3 P, const glm::vec3 PHit,
	const glm::vec3 PNorm, const glm::vec3 PHitNorm,
	const float r, const float rMax2, const float sigTr) 
{
	const float absVdotN = glm::abs(glm::dot(PNorm, PHitNorm));
	const float pdfdiskr = getPdfImportanceSampledDisk(r, sigTr, rMax2);
	const float RdHitDist = getSinglePlanarGaussianRd(glm::length(PHit - P), sigTr);
	return RdHitDist / (pdfdiskr*absVdotN);
}

__host__ __device__
float getA(const float eta) {
//calc diffuse fresnel reflection
	//donner page 41 fig 5.27
	//TODO: will always be less than one (1/ior) so no need for check?
	//just use jensen page 3 middle left
	const float fdr = eta < 1.f ?
		-0.4399f + 0.7099f / eta - 0.3319f / (eta*eta) + 0.0636f / (eta*eta*eta) :
		-1.4399f / (eta*eta) + 0.7099f / eta + 0.6681f + 0.0636f*eta;
	//jensen page 3 bottom left
	return (1.f + fdr) / (1.f - fdr);
}

__host__ __device__
void getRdJensen(glm::vec3& Rd, const Material& m, const float d2, const float eta) {//d2 is r^2
	const glm::vec3 sigA = m.sigA;
	const glm::vec3 sigSPrime = m.sigSPrime;
	//jensen page 2, below fig 2
	const glm::vec3 sigTPrime = sigA + sigSPrime;
	//jensen page 3 top left
	const glm::vec3 sigTr = glm::sqrt(3.f*sigA*sigTPrime);
	//jensen page 3 middle right
	const glm::vec3 zr = 1.f / sigTPrime;

	//doner page 43, paras above 5.33, fig 5.34
	//jensen page 2,3
	//zv = zr +4AD where D = 1/(3*sigTPrime) = zr / 3 since zr = 1/sigTPrime
	const float A = getA(eta);
	//jensen page 3middle right
	const glm::vec3 zv = zr *(1.f + 4.f/ 3.f * A);
	//dist from x to real source
	const glm::vec3 dr = glm::sqrt(zr*zr + d2);//finding hypot
	//dist from x to virtual source
	const glm::vec3 dv = glm::sqrt(zv*zv + d2);//finding hypot

	//jensen page 3 bot right, donner page 43
	const glm::vec3 alphaPrime = sigSPrime / sigTPrime;
	const glm::vec3 sigTrdv = sigTr * dv;
	const glm::vec3 sigTrdr = sigTr * dr;

	//TODO: will this go beyond 1]?
	Rd = 0.25f * InvPI * alphaPrime *
		(
		(zr * (1.f + sigTrdr) * glm::exp(-sigTrdr) / (dr*dr*dr)) +
		(zv * (1.f + sigTrdv) * glm::exp(-sigTrdv) / (dv*dv*dv))
		);
}

//NOTE: tangent space for sss can be arbitrary, unless UV maps are being used for 
//scattering params
__host__ __device__
void computeArbitraryTanToWorldUsingWorldNormal(const glm::vec3& nisect, 
	glm::mat3& tanToWorld) 
{
	tanToWorld[2] = nisect;
	const glm::vec3 T = glm::cross(glm::vec3(0, 1, 0), nisect);
	if (glm::length2(T) > 0.1f*0.1f) {
		tanToWorld[0] = glm::normalize(T);
		tanToWorld[1] = glm::normalize(glm::cross(nisect, T));
	} else {//world normal is basically 0,1,0
		tanToWorld[1] = glm::normalize(glm::cross(nisect, glm::vec3(1, 0, 0)));
		tanToWorld[0] = glm::normalize(glm::cross(tanToWorld[1], nisect));
	}
}

__host__ __device__
float getDiscSampleRay(Ray& discRay, glm::mat3& tanToWorld, const glm::vec3 nisect,
	const glm::vec3 pisect, thrust::default_random_engine& rng, 
	const float sigTr, const float rMax2, int& chosenAxis) 
{
	//see pbrtv3 pg910 for the diagram, just pythagorean for 'l'
	//we want to back off the disk along this dist to get ray's starting point
	//see king13 or pbrtv3 15.4 for the 3 axis disksample projection

	//glm::mat3 tanToObj;
	//tangent space to Object space transform, simply the obj space tbn vectors
	//computeTanToObjFromWorldNormalShape(g, nisect, tanToObj);
	//tanToWorld = g.invTranspose * tanToObj;
	//axes can be arbitrary since this is just for transforming a disk point
	computeArbitraryTanToWorldUsingWorldNormal(nisect, tanToWorld);

	
	glm::vec2 disksamp = getDiscXY(rng, sigTr, rMax2);
	const float r2 = glm::length2(disksamp);
	const float probeHalfLen = glm::sqrt(rMax2 - r2);//see king13 or pbrtv3, just pythag

	//must be transformed to world to keep things in the same scale (things are in mm)
	//if kept in obj space to save a transformation when projecting the disk ray back down
	//you would need to stretch the world space sample disk plane by the inverse transform
	//and in which case you're trading one transformation for another...
	thrust::uniform_real_distribution<float> u01(0, 1);
	const float unif = u01(rng);
	//for a chosen axis we back off along that axis by l/2 (probeHalfLen) then map our x y disk sample to the plane
	//that is orthogonal to the chosen axis, then orient the sample in the world with tanToWorld then move it to the
	//intersection point with pisect
	if (unif <= 0.5f) {//N (z)
	//if (unif <= 1.f) {//N (z)
		discRay.direction = -tanToWorld[2];
		discRay.origin = pisect + tanToWorld*glm::vec3(disksamp.x, disksamp.y, probeHalfLen);
		chosenAxis = 0;
	} else if (unif <= 0.75f) {//T (x)
		discRay.direction = -tanToWorld[0];
		discRay.origin = pisect + tanToWorld*glm::vec3(probeHalfLen, disksamp.x, disksamp.y);
		chosenAxis = 1;
	} else { //B (y)
		discRay.direction = -tanToWorld[1];
		discRay.origin = pisect + tanToWorld*glm::vec3(disksamp.y, probeHalfLen, disksamp.x);
		chosenAxis = 2;
	}
	return 2.f * probeHalfLen;//maximum ray trace t, any more than this and it wont contribute much anyway
}

__host__ __device__
float getMISWeight(const int chosenAxis, const glm::mat3& tanToWorld, 
	const float sigTr, const float rMax2, const glm::vec3& pisect,
	const glm::vec3& nPHit, const glm::vec3 pPHit, float& discPdf)
{
	//see pbrtv3 pg 799, ch14.3, 15.4
	//sample ratio is 2:1:1 so power heuristic is 2^2, 1^2, 1^2
	// so it will end up being4:1:1

	//we have 3 distributions to choose from
	//3 guassian disk projections along 3 axes. 
	//extending from pbrtv3 pg799 the formula is chosenDistributionPdfSquared / (sum of all distributions' squared pdfs)
	//where each each individual pdf is weighted by the number of samples taken in its distribution
	//do this before squaring
	//given that king13 says to choose the normal axis in tangent space
	//twice as often as the others our sampling ratio is 2:1:1
	//veach97 says that optimal beta is 2 which he found empircally
	//which is why we are squaring things

	//could save a calc if pushed only the necessary into ifs,code uglier
	//TODO: or just dont do the pdf calc in getDiskSampleRay
	//the rate at which they are selected needs to be taken into account, 
	//similar to picking bxdfs(think luminance (or regular) weighted glass bxdf selection)
	
	//calc pdfs, pPHit needs to be projected back on to the sampling disc plane
	//to get the correct r, "if this axis was the orginal sampling axis,
	//what would have been our randomly chosen r given how pPHit projects onto
	//this axis' disc sampling plane
	//pbrtv3 pg 913 must mult by axis prob
	const glm::vec3 Taxis = tanToWorld[0];
	const float pdfT = 0.25 * absDot(Taxis, nPHit) *
		getPdfImportanceSampledDisk_Reproject(pisect, pPHit, Taxis, sigTr, rMax2);

	const glm::vec3 Baxis = tanToWorld[1];
	const float pdfB = 0.25 * absDot(Baxis, nPHit) *
		getPdfImportanceSampledDisk_Reproject(pisect, pPHit, Baxis, sigTr, rMax2);

	const glm::vec3 Naxis = tanToWorld[2];
	const float pdfN = 0.5 * absDot(Naxis, nPHit) *
		getPdfImportanceSampledDisk_Reproject(pisect, pPHit, Naxis, sigTr, rMax2);

	if (0 == chosenAxis) {//chose the Normal axis in tangent space,z
		discPdf = pdfN;
		return powerHeuristic3(2, pdfN, 1, pdfT, 1, pdfB);
		//return 1.f;
	} else if(1 == chosenAxis) {//chose the Tangent axis, x
		discPdf = pdfT;
		return powerHeuristic3(1, pdfT, 2, pdfN, 1, pdfB);
	} else if(2 == chosenAxis) {//chose the BiTangent axis, y
		discPdf = pdfB;
		return powerHeuristic3(1, pdfB, 2, pdfN, 1, pdfT);
	}
}

__host__ __device__
float HenyeyGreensteinPhaseFunction(const glm::vec3& scatdir, const glm::vec3& dir, const float g)
{
	// g = -1 is full back scatter, 0 is isotropic, 1 is full forward scatter
	const float costheta = cosTheta(dir, scatdir);
	const float g2 = g*g;
	const float numer = 1 - g2;
	const float inside_denom = 1 + g2 - 2*g*costheta;
	const float cubed_denom = inside_denom*inside_denom*inside_denom;
	const float denom = glm::sqrt(cubed_denom);
	return Inv4PI * numer / denom;
}

__host__ __device__
glm::vec3 getL_S1(const Ray& pathray,
	const ShadeableIntersection& isect,  const int randlightindex, 
	const int numlights, thrust::default_random_engine& rng,
	const Material* materials,
	const Geom* dev_geoms, const int numgeoms, const int NUMSPLITS,
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices)
{
	////single scattering 
 //   //L(1) (integrating S(1)) jensen page 6
	const glm::vec3 wo = -pathray.direction;
	const glm::vec3 pisect = getPointOnRay(pathray, isect.t) + isect.surfaceNormal*EPSILON;
	const glm::vec3 nisect = isect.surfaceNormal;
	const Material& m = materials[isect.materialId];
	const glm::vec3 Ft_wo = 1.f - evaluateFresnelDielectric(cosTheta(nisect,wo), 1, m.indexOfRefraction);
	
	//sigT, sigTc, sigS, G, g //defs scattered in jensen01
	const float g = 0;//isotropic (close to 1 is forward close to -1 backward)
	//g is the average costheta between wi and -wo during 
	//a scattering event inside the material 
	// > 0 means you tend to scatter forward and < 0 means backward
	const glm::vec3 sigS = m.sigSPrime / (1 - g);
	const glm::vec3 sigT = m.sigA + sigS;
	const float sigTLum = getLuminance(sigT);

	const glm::vec3 refracted = glm::refract(-wo, nisect, 1 / m.indexOfRefraction);//glm expects incoming ray

	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 L_S1(0);
	for (int i = 0; i < NUMSPLITS; ++i) {
		//find single scatter point
		glm::vec3 psurface; glm::vec3 nsurface;
		float pdfdirect = 0; glm::vec3 widirect(0.f); glm::vec3 colordirect(0.f);
		float soPrime; float singlepdf; glm::vec3 psingle;
		glm::vec3 plightsamp; glm::vec3 nlightsamp;
		bool foundSample = false; int count = 0;

		while (!foundSample) {
			soPrime = getSingleScatterDist(rng, sigTLum);
			singlepdf = getSingleScatterPdf(soPrime, sigTLum);
			psingle = pisect + soPrime*refracted;
			//select light
			const Geom& randlight = dev_geoms[randlightindex];
			const Material& mlight = materials[randlight.materialid];

			//hacky thing, colordirect will be incoming irradiance
			//NOTE: nisect could prob be replaced with an appropriate rotated version 
			//rotated by costheta(-wo, refracted) about normalize(cross(-wo, refracted))
			colordirect = sampleLight(randlight, mlight, psingle,
				nisect, rng, widirect, pdfdirect, plightsamp, nlightsamp);
			if (isBlack(colordirect) || 0 >= pdfdirect) { continue; }
			pdfdirect /= numlights;

			//spawn ray from isect point and use widirect for direction 
			Ray wiray_direct;
			wiray_direct.origin = psingle;
			wiray_direct.direction = widirect;

			//poke through first
			const Geom& thisgeom = dev_geoms[isect.geomId];
			bool outside;
			const float tsurface = shapeIntersectionTest(thisgeom, wiray_direct, psurface, nsurface, outside, dev_BVHNodes, dev_TriIndices, dev_TriVertices);
			if (thisgeom.type == GeomType::CUBE && glm::dot(nsurface, widirect) < 0.f)	nsurface *= -1;//cube is flipping the normal other shapes aren't
			psurface += nsurface*0.005f;//need a little more umf than EPSILON
			//if (0 > tsurface || outside == true) { continue; }//if our scatter length is outside? prob set a point below the surface and shoot in the direction of refracted and see what tmax is
			if (0 <= tsurface && outside == false) { foundSample = true; }
			if (100 <= count++) { break; }
		}
		if (!foundSample) { continue; }

		//get the closest intersection in scene
		//Ray surfaceray; surfaceray.direction = widirect; surfaceray.origin = psurface;
		//NOTE: we moved our ray origin from psingle to psurface need to reset direction as we may miss this time.
		//we want to see if light can see this point on the surface
		Ray surfaceray; surfaceray.direction = glm::normalize((plightsamp - nlightsamp*0.05f) - psurface); surfaceray.origin = psurface;
		int hitgeomindex = -1;
		findClosestIntersection(surfaceray, dev_geoms, numgeoms, hitgeomindex, dev_BVHNodes, dev_TriIndices, dev_TriVertices);

		//TODO: prob only need first and last condition
		if (hitgeomindex != randlightindex) { continue; }

		const float absdotdirect = absDot(nsurface, widirect);

		//NOTE: jensen page 4 fig 6 does not have ndotwi included for L1
		//also not there in the equation on page 6 from which it is derived
		colordirect = colordirect / pdfdirect;
		const glm::vec3 Ft_wi = 1.f - evaluateFresnelDielectric(absdotdirect, 1, m.indexOfRefraction);
		const float hgphase_p = HenyeyGreensteinPhaseFunction(widirect,refracted,g);
		//G = absDot(norm_xi, wo)/absDot(norm_xi,wi)//NOTE: the paper mentions that wo and wi are the refracted
		//incoming and outgoing directions for single scattering i.e. wo is the refracted -wo and wi is the 
		//direction towards the light
		const float G = absDot(nsurface, refracted) / absdotdirect;
		const glm::vec3 sigTc = sigT + G*sigT;//NOTE: the paper mentions that first sigT is for xo and the second is xi
		//jensen big equation bottom page 6, si is observed distance and siprime is refracted distance
		//if the light had actually refracted through to psingle
		const float si = glm::length(psurface-psingle);
		const float inveta = m.indexOfRefraction;
		//NOTE: siPrime will be NAN if light source is close by. Sad. Jensen says this equation assumes light is far away.
		//try clamping denom to near 0 for the calc of siprime if light is close and you hit that NAN case, looks ok, not sure if it looks 'correct'
		const float siPrime_denom = 1.f - inveta*inveta*(1.f - absdotdirect*absdotdirect);
		//const float siPrime = siPrime_denom > 0 ? si * absdotdirect / glm::sqrt(siPrime_denom) : si*absdotdirect / 0.01; 
		const float siPrime = si;
		const glm::vec3 F = Ft_wo*Ft_wi;
		//all sigmas are position dependent (if you do textures these can be set by them)
		const glm::vec3 L_S1sample = (sigS * F * hgphase_p)*(1.f / sigTc)*
			glm::exp(-siPrime*sigT)*glm::exp(-soPrime*sigT)*colordirect / (singlepdf);//light pdf included in colordirect

		L_S1 += L_S1sample;
	}
	return L_S1 * (1.f / NUMSPLITS);
}

__host__ __device__
glm::vec3 getL_Sd(const Ray& pathray, const ShadeableIntersection& isect, 
	const int randlightindex, const int numlights, 
	thrust::default_random_engine& rng, const Material* materials, 
	const Geom* dev_geoms, const int numgeoms, const int NUMSPLITS,
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices)
{
	//Sd jensen page 3
	//we may want to perform splitting if the frame time isnt too bad,
	//to converge faster for this noisy material
	//(pbrtv3 13.7, simply just evaluate multiple times and take average) 

	const glm::vec3 wo = -pathray.direction;
	const glm::vec3 pisect = getPointOnRay(pathray, isect.t) + isect.surfaceNormal*EPSILON;
	const glm::vec3 nisect = isect.surfaceNormal;
	const Material& m = materials[isect.materialId];
	const glm::vec3 Ft_wo = 1.f - evaluateFresnelDielectric(cosTheta(nisect, wo), 1, m.indexOfRefraction);
	const glm::vec3 sigTPrime = m.sigA + m.sigSPrime;
	const glm::vec3 sigTr = glm::sqrt(3.f*m.sigA*sigTPrime);
	const float sigTrLum = getLuminance(sigTr);//could do max but lum is a better metric, should prob do all 3 channels in future
	const float v = 1.f / (2.f * sigTrLum);//sigTr = 1/(2v)//looking at king13 vs jensen01 notation
	const float rMax = glm::sqrt(v * 12.46f);//Rm from King13, but we dont divide basically we dont care about the outskirts of the gaussian disk
	const float rMax2 = rMax*rMax;

	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 L_Sd(0);//multiscatter component of S from jensen01
	for (int i = 0; i < NUMSPLITS; ++i) {
		int chosenAxis;
		glm::mat3 tanToWorld;
		bool foundSample = false; int count = 0;
		glm::vec3 pPHit; glm::vec3 nPHit;
		while (!foundSample) {
			Ray discRay;

			const float max_t = getDiscSampleRay(discRay, tanToWorld, nisect, pisect,
				rng, sigTrLum, rMax2, chosenAxis);

			const Geom& thisgeom = dev_geoms[isect.geomId];
			bool outside;
			const float t = shapeIntersectionTest(thisgeom, discRay, pPHit, nPHit, outside, dev_BVHNodes, dev_TriIndices, dev_TriVertices);
			if (thisgeom.type == GeomType::CUBE && !outside) { nPHit *= -1.f; } //cube's flipping the normal, other shapes aren't
			pPHit += nPHit*EPSILON;

			//if (t > max_t || t < 0) { continue; } 
			if (t <= max_t && t >= 0) { foundSample = true; } 
			if (100 <= count++) { break; }//dim/inaccurate better than inf loop
		}
		if (!foundSample) { continue; }

		//we hit, call Rd from jensen01
		glm::vec3 RdJensen;
		getRdJensen(RdJensen, m, glm::length2(pPHit - pisect), 1.f / m.indexOfRefraction);

		float pdfdirect = 0; glm::vec3 widirect(0.f); glm::vec3 colordirect(0.f);

		const Geom& randlight = dev_geoms[randlightindex];
		const Material& mlight = materials[randlight.materialid];

		//hacky thing, colordirect will be incoming irradiance
		glm::vec3 plightsamp; glm::vec3 nlightsamp;
		colordirect = sampleLight(randlight, mlight, pPHit,
			nPHit, rng, widirect, pdfdirect, plightsamp,nlightsamp);
		if (0 >= pdfdirect || isBlack(colordirect)) { continue; }
		pdfdirect /= numlights;

		//spawn ray from isect point and use widirect for direction 
		Ray wiray_direct; wiray_direct.direction = glm::normalize((plightsamp-nlightsamp*0.05f) - pPHit); wiray_direct.origin = pPHit;
		//get the closest intersection in scene
		int hitgeomindex = -1;
		findClosestIntersection(wiray_direct, dev_geoms, numgeoms, hitgeomindex, dev_BVHNodes, dev_TriIndices, dev_TriVertices);

		if (hitgeomindex != randlightindex) { continue; }
		const float absdotdirect = absDot(widirect, nPHit);
		colordirect = colordirect * absdotdirect / pdfdirect;

		const glm::vec3 Ft_wi = 1.f - evaluateFresnelDielectric(cosTheta(nPHit, widirect), 1, m.indexOfRefraction);

		float discSamplePdf;
		const float misweight = getMISWeight(chosenAxis, tanToWorld, sigTrLum, rMax2, pisect,
			nPHit, pPHit, discSamplePdf);

		//jensen01 pg3 fig5
		//colordirect contains the DL sampling pdf
		L_Sd += (misweight * InvPI * Ft_wi * RdJensen * Ft_wo * colordirect) / discSamplePdf;

	}
	return L_Sd * (1.f / NUMSPLITS);
}

__host__ __device__
glm::vec3 getBSSRDF_DL(const PathSegment& path, 
	const ShadeableIntersection& isect, const Material* materials, 
	thrust::default_random_engine& rng, const int randlightindex,
	const int numlights, const Geom* dev_geoms, const int numgeoms,
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices
) 
{
	const int NUMSPLITS = 10; 

	const glm::vec3 multiscatterTerm = getL_Sd(path.ray, isect, randlightindex,
		numlights, rng, materials, dev_geoms, numgeoms, NUMSPLITS, dev_BVHNodes, dev_TriIndices, dev_TriVertices);
	
	const glm::vec3 singlescatterTerm = getL_S1(path.ray, isect, randlightindex,
		numlights, rng, materials, dev_geoms, numgeoms, NUMSPLITS, dev_BVHNodes, dev_TriIndices, dev_TriVertices);

	return multiscatterTerm + singlescatterTerm;
	//return multiscatterTerm;
	//return singlescatterTerm;
}

__host__ __device__
void chooseSubSurface(PathSegment& path, ShadeableIntersection& isect, const Material& m,
	float& bxdfPDF, glm::vec3& bxdfColor, thrust::default_random_engine& rng, const Geom* dev_geoms, 
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices )
{
	//disk sample until we get a hit
	const glm::vec3 wo = -path.ray.direction;
	const glm::vec3 pisect = getPointOnRay(path.ray, isect.t) + isect.surfaceNormal*EPSILON;
	const glm::vec3 nisect = isect.surfaceNormal;
	const glm::vec3 Ft_wo = 1.f - evaluateFresnelDielectric(cosTheta(nisect, wo), 1, m.indexOfRefraction);
	const glm::vec3 sigTPrime = m.sigA + m.sigSPrime;
	const glm::vec3 sigTr = glm::sqrt(3.f*m.sigA*sigTPrime);
	const float sigTrLum = getLuminance(sigTr);//could do max but lum is a better metric, should prob do all 3 channels in future
	const float v = 1.f / (2.f * sigTrLum);//sigTr = 1/(2v)//looking at king13 vs jensen01 notation
	const float rMax = glm::sqrt(v * 12.46f);//Rm from King13, but we dont divide basically we dont care about the outskirts of the gaussian disk
	const float rMax2 = rMax*rMax;

	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 L_Sd(0);//multiscatter component of S from jensen01
	int chosenAxis; glm::mat3 tanToWorld;
	glm::vec3 pPHit(FLT_MAX); glm::vec3 nPHit; 
	bool foundSample = false; int count = 0;
	while (!foundSample) {
		Ray discRay;
		const float max_t = getDiscSampleRay(discRay, tanToWorld, nisect, pisect,
			rng, sigTrLum, rMax2, chosenAxis);

		//testintersection against isect's geometry(store geom index)
		//intersection should return a new intersection within max_t with a pos and normal
		const Geom& thisgeom = dev_geoms[isect.geomId];
		bool outside;
		const float t = shapeIntersectionTest(thisgeom, discRay, pPHit, nPHit, outside, dev_BVHNodes, dev_TriIndices, dev_TriVertices);
		if (thisgeom.type == GeomType::CUBE && !outside) { nPHit *= -1.f; } //cube's flipping the normal, other shapes aren't
		pPHit += nPHit*0.005f;

		if (t <= max_t && t >= 0) { foundSample = true; } 
		if (100 <= count++) {
			bxdfColor = glm::vec3(0); bxdfPDF = 0;
			return;
		}
	}

	//we hit, call Rd from jensen01
	glm::vec3 RdJensen;
	getRdJensen(RdJensen, m, glm::length2(pPHit - pisect), 1.f / m.indexOfRefraction);

	const glm::vec3 wiBxdfSample = calculateRandomDirectionInHemisphere(nisect, rng);

	const glm::vec3 Ft_wi = 1.f - evaluateFresnelDielectric(cosTheta(nPHit, wiBxdfSample), 1, m.indexOfRefraction);

	float discSamplePdf;
	const float misweight = getMISWeight(chosenAxis, tanToWorld, sigTrLum, rMax2, pisect,
		nPHit, pPHit, discSamplePdf);

	//colordirect contains the DL sampling pdf
	//jensen01 pg3 fig5
	//record the pdf and color
	const float CosSampledDirBxdfPDF = sameHemisphere(wo, wiBxdfSample, nPHit) ? cosTheta(nPHit, wiBxdfSample)*InvPI : 0.f;
	bxdfColor = (misweight * InvPI * Ft_wi * RdJensen * Ft_wo);
	bxdfPDF = discSamplePdf * CosSampledDirBxdfPDF;

	//update path with the new xi and wi
	isect.surfaceNormal = nPHit;
	path.ray.origin = pPHit;
	path.ray.direction = wiBxdfSample;
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
	ShadeableIntersection& isect,
	const Material& m,
	float& bxdfPDF,
	glm::vec3& bxdfColor,
	thrust::default_random_engine &rng, const Geom* dev_geoms, 
	const BVHNode* dev_BVHNodes, const glm::ivec3* dev_TriIndices, const Vertex* dev_TriVertices )
{
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

	float avepdf = 0.f;
	glm::vec3 avecolor(0.f);


	 if (1 == m.hasSubSurface) {//subsurface
		thrust::uniform_real_distribution<float> u01(0, 1);

		//BAD, fireflies galore
		//float SchlickR = 0.f;
		//if (1 == m.hasReflective) { 
		//	float SchlickR0 = (1.f - m.indexOfRefraction) / (1.f + m.indexOfRefraction);
		//	SchlickR0 *= SchlickR0;
		//	const float cosi = cosTheta(isect.surfaceNormal, -path.ray.direction);
		//	//this will equal 1 when light heads straight into surface
		//	//in that case should be prob that we do a transmission interaction and not a reflection
		//	SchlickR = SchlickR0 + (1.f - SchlickR0)*cosi*cosi*cosi*cosi*cosi;
		//	SchlickR = 1.f - SchlickR;//get refl prob proxy
		//}
		//if (u01(rng) < SchlickR) {
		//	chooseReflection(path, isect, m, bxdfPDF, bxdfColor, rng, true, SchlickR);
		//	path.specularbounce = true;
		//} else {
		//	chooseSubSurface(path, isect, m, bxdfPDF, bxdfColor, rng, dev_geoms);
		//	bxdfPDF *= (1.f - SchlickR);
		//}

		const float prob = (1 == m.hasReflective) ? 0.5f : 0.f;
		if (u01(rng) < prob) {
			chooseReflection(path, isect, m, bxdfPDF, bxdfColor, rng, true, prob);
			path.specularbounce = true;
		} else {
			chooseSubSurface(path, isect, m, bxdfPDF, bxdfColor, rng, dev_geoms, 
				dev_BVHNodes, dev_TriIndices, dev_TriVertices);
			bxdfPDF *= (1.f - prob);
		}
     } else if (0 == m.hasReflective && 0 == m.hasRefractive) {//just diffuse
		 path.ray.origin = getPointOnRay(path.ray, isect.t);
		const glm::vec3 normal = isect.surfaceNormal;
		const glm::vec3 wo = -path.ray.direction;
		path.ray.origin += normal*EPSILON;
		path.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		const glm::vec3 wi = path.ray.direction;
		bxdfPDF = sameHemisphere(wo, wi,normal) ? cosTheta(normal, wi)*InvPI : 0.f;
		bxdfColor = m.color * InvPI;
	} else if (1 == m.hasReflective && 0 == m.hasRefractive) {//just reflective
		chooseReflection(path, isect, m, bxdfPDF, bxdfColor, rng, false, 1.f);
	} else if (0 == m.hasReflective &&  1 == m.hasRefractive) {//just transmissive
		chooseTransmission(path, isect, m, bxdfPDF, bxdfColor, rng, false, 0.f);
	} else if (1 == m.hasReflective &&  1 == m.hasRefractive) {//relective and transmissive
		//determine reflect probability based on luminance 
		const glm::vec3 colorR = m.color;
		const glm::vec3 colorT = m.specular.color;
		const float colorRLum = getLuminance(colorR);
		const float colorTLum = getLuminance(colorT);
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
