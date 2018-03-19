#pragma once

#include <cuda.h>
#include <curand.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>


#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/component_wise.hpp"
#include "utilities.h"
#include "utilkern.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

////////////////////////////////////////////////////
////////////           NAIVE            ////////////
////////////////////////////////////////////////////
__global__ void shadeMaterialNaive (
  const int iter, const int num_paths, 
	const ShadeableIntersection * shadeableIntersections, 
	PathSegment * pathSegments, const Material * materials, const Geom* dev_geoms
	)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_paths) { return; }

	PathSegment& path = pathSegments[idx];
	//const ShadeableIntersection& isect = shadeableIntersections[idx];
	ShadeableIntersection isect = shadeableIntersections[idx];

#if 0 == COMPACT
	if (0 >= path.remainingBounces) { return; }
#endif

	//Hit Nothing
	if (0.f >= isect.t) {// Lots of renderers use 4 channel color, RGBA, where A = alpha, often // used for opacity, in which case they can indicate "no opacity".  // This can be useful for post-processing and image compositing.
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
		return;
	}

	const Material& material = materials[isect.materialId];

	//Hit Light
	if (0.f < material.emittance) {
		path.color *= (material.color * material.emittance); 
		path.remainingBounces = 0;
		return;
	}

	//this was last chance to hit light
	if (0 >= --path.remainingBounces) {
		path.color = glm::vec3(0.f);
		path.remainingBounces = 0;
		return;
	}

	//Hit Material, generate new ray for the path(wi), get pdf and color for the material, use those to mix with the path's existing color
	float bxdfPDF; glm::vec3 bxdfColor;
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
	scatterRayNaive(path, isect, material, bxdfPDF, bxdfColor, rng, dev_geoms);
	if (0 >= bxdfPDF || isBlack(bxdfColor)) {
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
		return;
	}
	path.color *= bxdfColor * absDot(isect.surfaceNormal, path.ray.direction) / bxdfPDF;
}


/////////////////////////////////////////////////////////////
////////////MULTIKERN MULTIPLE IMPORTANCE SAMPLING///////////
////////////////////////////////////////////////////////////
__global__ void shadeMaterialMIS_DLlight(const int iter, const int depth,
	const int numlights, const int MAXBOUNCES,
	const int num_paths, const ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments, const Material* materials, 
	const int* dev_geomLightIndices, const Geom* dev_geoms, const int numgeoms) 
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int terminatenumber = -100;

	if (idx >= num_paths) { return; }
	PathSegment& path = pathSegments[idx];
	const ShadeableIntersection& isect = shadeableIntersections[idx];
	
//#if 0 == COMPACT
	if (terminatenumber >= path.remainingBounces) { return; }
//#endif

	//Hit Nothing
	if (0.f >= isect.t) {
		path.remainingBounces = terminatenumber;
		return;
	}

	const Material& m = materials[isect.materialId];

	//First Bounce or Last bounce was specular
	//NOTE: may want to remove spec bounce check as the matchesflags from 561 didnt work
	const glm::vec3 wo = -path.ray.direction;
	if (0 == depth || path.specularbounce) { //what if path color started at 1 and you multiplied as you went along???
		path.color += path.throughput * Le(m, isect.surfaceNormal, wo); 
	}

	//Hit Light
	if (0.f < m.emittance) {
		path.remainingBounces = terminatenumber;//already accounted for Le above
		return;
	}

	path.specularbounce = (m.hasReflective || m.hasRefractive) ? 1 : 0;
	if (path.specularbounce) { return; } 

	//SKIP IF SSS FOR NOW
	if (1 == m.hasSubSurface) { return; }


	//////////////////////
	//////  DIRECT  //////
	//////////////////////
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);
	float pdfdirect = 0;
	glm::vec3 widirect(0.f);
	glm::vec3 colordirect(0.f);

	const int randlightindex = dev_geomLightIndices[int( u01(rng)*numlights )];
	const Geom& randlight = dev_geoms[randlightindex];
	const Material& mlight = materials[randlight.materialid];
	const glm::vec3 pisect = getPointOnRay(path.ray, isect.t) + isect.surfaceNormal*EPSILON;

	///////////////////////
	///////DEBUG///////////
	//////////////////////
	//int pixelx = path.MSPaintPixel.x;
	//int pixely = path.MSPaintPixel.y;
	//if (400 == pixelx && 580 == pixely && 0 == depth) {//black spec debug mode, top of box 
	//	printf("\npixelx: %i, pixely: %i, depth: %i, iter: %i", pixelx, pixely, depth, iter);
	//}

	////////////////////////
	/////// LIGTH SAMPLE////
	////////////////////////
	//sample the light to get widirect, pdfdirect, and colordirect
	glm::vec3 plightsamp;
	glm::vec3 nlightsamp;
	colordirect = sampleLight(randlight, mlight, pisect,
		isect.surfaceNormal, rng, widirect, pdfdirect, plightsamp, nlightsamp);
	pdfdirect /= numlights;

	//spawn ray from isect point and use widirect for direction 
	Ray wiray_direct;
	wiray_direct.origin = pisect;
	wiray_direct.direction = widirect;

	//get the closest intersection in scene
	int hitgeomindex = -1;
	findClosestIntersection(wiray_direct, dev_geoms, numgeoms, hitgeomindex);

	//TODO: prob only need first and last condition
	if (0 >= pdfdirect || 0 > hitgeomindex || hitgeomindex != randlightindex) {
		return;
	} else if (0.f < pdfdirect) {//condition needed? is OR async on gpu?
		glm::vec3 bxdfcolordirect; float bxdfpdfdirect;
		bxdf_FandPDF(wo, widirect, isect.surfaceNormal, 
			m, bxdfcolordirect, bxdfpdfdirect); 
		const float absdotdirect = absDot(widirect, isect.surfaceNormal);
		const float powerheuristic_direct = powerHeuristic(1,pdfdirect,1,bxdfpdfdirect);
		colordirect = (bxdfcolordirect * colordirect * absdotdirect * powerheuristic_direct) / pdfdirect;
	}
	path.color += path.throughput*colordirect;
}

/////////////////////////////////////////////////////////////
////////////MULTIKERN MULTIPLE IMPORTANCE SAMPLING///////////
////////////////////////////////////////////////////////////
__global__ void shadeMaterialMIS_DLbxdf(const int iter, const int depth,
	const int numlights, const int MAXBOUNCES,
	const int num_paths, const ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments, const Material* materials,
	const int* dev_geomLightIndices, const Geom* dev_geoms, const int numgeoms) 
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int terminatenumber = -100;

	if (idx >= num_paths) { return; }
	PathSegment& path = pathSegments[idx];
	if (path.specularbounce) { return; }
	if (terminatenumber >= path.remainingBounces) { return; }
	//const ShadeableIntersection& isect = shadeableIntersections[idx];
	ShadeableIntersection isect = shadeableIntersections[idx];
	const Material& m = materials[isect.materialId];

	//SKIP IF SSS FOR NOW
	if (1 == m.hasSubSurface) { return; }


	/////////////////////////
	/////////DEBUG///////////
	////////////////////////
	//int pixelx = path.MSPaintPixel.x;
	//int pixely = path.MSPaintPixel.y;
	//if (400 == pixelx && 580 == pixely && 0 == depth) {//black spec debug mode, top of box 
	//	printf("\npixelx: %i, pixely: %i, depth: %i, iter: %i", pixelx, pixely, depth, iter);
	//}
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);
	const int randlightindex = dev_geomLightIndices[int( u01(rng)*numlights )];
	const Geom& randlight = dev_geoms[randlightindex];
	const glm::vec3 pisect = getPointOnRay(path.ray, isect.t) + isect.surfaceNormal*EPSILON;
	///////////////////////////////////////////////////////////////
	////////// SAMPLE BXDF, CHECK IF HIT A LIGHT //////////////////
	///////////////////////////////////////////////////////////////
	glm::vec3 colordirectsample(0);
	glm::vec3 widirectsample;
	float pdfdirectsample;
	PathSegment pathcopy = path;//scatterRayNaive updates the path with isect origin and wi direction
	scatterRayNaive(pathcopy, isect, m, pdfdirectsample, colordirectsample, rng, dev_geoms);
	widirectsample = pathcopy.ray.direction;
	if (glm::length2(colordirectsample) > 0 && pdfdirectsample > 0) {
		float DLPdf_for_directsample = lightPdfLi(randlight, pisect, widirectsample);

		if (DLPdf_for_directsample > 0) {//widirectsample can hit it (might not be closest though)
			Ray widirectsampleRay; widirectsampleRay.direction = widirectsample; widirectsampleRay.origin = pisect;
			int posslightindex;//we know a our light is in the widirectsample direction, lets try to hit it, and hopefully its the closest light
			const glm::vec3 nposslight = findClosestIntersection(widirectsampleRay, dev_geoms, numgeoms, posslightindex);

			if (posslightindex == randlightindex) {//only want one light per direct lighting sample pair(need to sample both a rand light and the material bxdf to cover edge cases of combos of light size/intensity and bxdf)
				DLPdf_for_directsample /= numlights;
				const float powerheuristic_directsample = powerHeuristic(1, pdfdirectsample, 1, DLPdf_for_directsample);
				const float absdotdirectsample = absDot(widirectsample, isect.surfaceNormal);
				const Geom& posslight = dev_geoms[posslightindex];
				const Material& mposslight = materials[posslight.materialid];
				const glm::vec3 Li = Le(mposslight, nposslight, -widirectsample);//returns 0 if we don't hit a light(no emmisive)
				colordirectsample = (colordirectsample * Li * absdotdirectsample * powerheuristic_directsample) / pdfdirectsample;
			} else {
				colordirectsample = glm::vec3(0);
			}
		} else {
			colordirectsample = glm::vec3(0);
		}
	} else {//prob dont need 
		colordirectsample = glm::vec3(0);
	}

	path.color += path.throughput*colordirectsample;

}

/////////////////////////////////////////////////////////////
////////////MULTIKERN MULTIPLE IMPORTANCE SAMPLING///////////
////////////////////////////////////////////////////////////
__global__ void shadeMaterialMIS_throughput(const int iter, const int depth,
	const int numlights, const int MAXBOUNCES,
	const int num_paths, const ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments, const Material* materials,
	const int* dev_geomLightIndices, const Geom* dev_geoms, const int numgeoms) 
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int terminatenumber = -100;

	if (idx >= num_paths) { return; }
	PathSegment& path = pathSegments[idx];
	if (terminatenumber >= path.remainingBounces) { return; }
	//const ShadeableIntersection& isect = shadeableIntersections[idx];
	ShadeableIntersection isect = shadeableIntersections[idx];
	const Material& m = materials[isect.materialId];
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);

	////////////////////////////
	///////////DEBUG////////////
	////////////////////////////
	int pixelx = path.MSPaintPixel.x;
	int pixely = path.MSPaintPixel.y;
	//if (398 == pixelx && 580 == pixely && 0 == depth) {//black spec debug mode, top of box 
	//if (400 == pixelx && 580 == pixely && 0 == depth) {//black spec debug mode, top of box 
	//	printf("\npixelx: %i, pixely: %i, depth: %i, iter: %i", pixelx, pixely, depth, iter);
	//}

	//DO SSS HERE FOR NOW
	if (1 == m.hasSubSurface) {
	const int randlightindex = dev_geomLightIndices[int( u01(rng)*numlights )];
		path.color += path.throughput*getBSSRDF_DL(path, isect, materials, rng, 
			randlightindex, numlights, dev_geoms, numgeoms);
		//global ray will just be cos sampled dir from the last Sd disk sample
		path.remainingBounces = terminatenumber;
		return;
	}



	//////////////////////////
	////////  INDIRECT  //////
	//////////////////////////
	////get global illum ray from bxdf and loop
	////do a normal samplef to get wi pdf and color
	//float bxdfPDF; glm::vec3 bxdfColor;
	//scatterRayNaive(path, isect, m, bxdfPDF, bxdfColor, rng, dev_geoms);
	//const glm::vec3 bxdfWi = path.ray.direction;
	//const glm::vec3 normal = isect.surfaceNormal;

	//glm::vec3 current_throughput(0.f);
	//if (!isBlack(bxdfColor) && 0 < bxdfPDF) {
	//	current_throughput = bxdfColor * absDot(bxdfWi, normal) / bxdfPDF;
	//}
	//path.throughput *= current_throughput;

	////russian roulette terminate low-energy rays
	if (depth > MAXBOUNCES) {
		const float max = glm::compMax(path.throughput);
		if (max < u01(rng)) {
			path.remainingBounces = terminatenumber;
			return;
		}
		//in the off chance this ray is still going after a long time
		//scale it up to reduce noise 
		//i.e. make it more like the rays in its pixel who terminated earlier
		//this one just got lucky presumably
		path.throughput /= max;
	}
}


__host__ __device__
void MIS_BxdfSampleContribuition(glm::vec3& colordirectsample, const glm::vec3& colordirect,
	const glm::vec3& pisect, ShadeableIntersection& isect, 
	const PathSegment& path, const Material& m, 
	const Geom& randlight, const int randlightindex, const int numlights,
	thrust::default_random_engine& rng, const Material* materials,
	const Geom* dev_geoms, const int numgeoms
) 
{
	if (1 == path.specularbounce) { return; }
	if (1 == m.hasSubSurface) { return; } //NOTE: will need this if m also has microfacet
	glm::vec3 widirectsample;
	float pdfdirectsample;
	PathSegment pathcopy = path;//scatterRayNaive updates the path with isect origin and wi direction, so make copy
	scatterRayNaive(pathcopy, isect, m, pdfdirectsample, colordirectsample, rng, dev_geoms);
	widirectsample = pathcopy.ray.direction;
	////////////REMOVE isBlack CHECK///////////////////////
	if (0 >= glm::length2(colordirectsample) || 0 >= pdfdirectsample || !isBlack(colordirect)) {//only take if we got nothing for surface sampling part
		colordirectsample = glm::vec3(0); 
		return;
	}

	float DLPdf_for_directsample = lightPdfLi(randlight, pisect, widirectsample);

	//widirectsample can hit it (might not be closest though)
	if (0 >= DLPdf_for_directsample) {
		colordirectsample = glm::vec3(0); 
		return;
	}

	Ray widirectsampleRay; widirectsampleRay.direction = widirectsample; widirectsampleRay.origin = pisect;
	int posslightindex;//we know a our light is in the widirectsample direction, lets try to hit it, and hopefully its the closest light
	const glm::vec3 nposslight = findClosestIntersection(widirectsampleRay, dev_geoms, numgeoms, posslightindex);

	//only want one light per direct lighting sample pair
	//need to sample both a rand light and the material bxdf
	//to cover edge cases of combos of light size/intensity and bxdf
	if (posslightindex != randlightindex) {
		colordirectsample = glm::vec3(0); 
		return;
	}

	DLPdf_for_directsample /= numlights;
	const float powerheuristic_directsample = powerHeuristic(1, pdfdirectsample, 1, DLPdf_for_directsample);
	const float absdotdirectsample = absDot(widirectsample, isect.surfaceNormal);
	const Geom& posslight = dev_geoms[posslightindex];
	const Material& mposslight = materials[posslight.materialid];
	const glm::vec3 Li = Le(mposslight, nposslight, -widirectsample);//returns 0 if we don't hit a light(no emmisive)
	colordirectsample = (colordirectsample * Li * absdotdirectsample * powerheuristic_directsample) / pdfdirectsample;
}

__host__ __device__
void MIS_LightSampleContribution(glm::vec3& colordirect, const ShadeableIntersection& isect, 
	const glm::vec3& pisect, const glm::vec3& nisect, 
	const glm::vec3& wo, const PathSegment& path, const Material& m, 
	const Geom& randlight, const int randlightindex, const int numlights,
	thrust::default_random_engine& rng, const Material* materials,  
	const Geom* dev_geoms, const int numgeoms
) 
{
	if (1 == path.specularbounce) { return; }
	float pdfdirect = 0;
	glm::vec3 widirect(0.f);
	const Material& mlight = materials[randlight.materialid];
	//sample the light to get widirect, pdfdirect, and colordirect
	//if BSSRDF then need xi and ni, xo and no is where our ray first hit
	if (1 == m.hasSubSurface) {
		//just do single and multi scatter direct lighting here
		colordirect = getBSSRDF_DL(path, isect, materials, rng, randlightindex, numlights, dev_geoms, numgeoms);
		return;
	}
	glm::vec3 plightsamp; glm::vec3 nlightsamp;
	colordirect = sampleLight(randlight, mlight, pisect,
		nisect, rng, widirect, pdfdirect, plightsamp, nlightsamp);
	if (0 >= pdfdirect || isBlack(colordirect)) { colordirect = glm::vec3(0); return; }
	pdfdirect /= numlights;

	//spawn ray from isect point and use widirect for direction 
	Ray wiray_direct;
	wiray_direct.origin = pisect;
	wiray_direct.direction = widirect;

	//get the closest intersection in scene
	int hitgeomindex = -1;
	findClosestIntersection(wiray_direct, dev_geoms, numgeoms, hitgeomindex);
	if (hitgeomindex != randlightindex) { 
		colordirect = glm::vec3(0); 
		return; 
	}

	glm::vec3 bxdfcolordirect; float bxdfpdfdirect;
	bxdf_FandPDF(wo, widirect, nisect, m, bxdfcolordirect, bxdfpdfdirect);
	const float absdotdirect = absDot(widirect, nisect);
	const float powerheuristic_direct = powerHeuristic(1, pdfdirect, 1, bxdfpdfdirect);
	colordirect = (bxdfcolordirect * colordirect * absdotdirect * powerheuristic_direct) / pdfdirect;
}


///////////////////////////////////////////////////
////////////MULTIPLE IMPORTANCE SAMPLING///////////
///////////////////////////////////////////////////
__global__ void shadeMaterialMIS(const int iter, const int depth,
	const int numlights, const int MAXBOUNCES,
	const int num_paths, const ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments, const Material* materials, 
	const int* dev_geomLightIndices, const Geom* dev_geoms, const int numgeoms) 
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int terminatenumber = -100;

	if (idx >= num_paths) { return; }
	PathSegment& path = pathSegments[idx];
	//const ShadeableIntersection& isect = shadeableIntersections[idx];
	ShadeableIntersection isect = shadeableIntersections[idx];
	
	/////////////
	////DEBUG////
	/////////////
	//if (idx == 0) { 
		//printf("\nhello %i", idx); //you can printf from the gpu!
	//}
	//int pixelx = path.MSPaintPixel.x;
	//int pixely = path.MSPaintPixel.y;
	//if (270 == pixelx && 220 == pixely) {
	//	printf("\npixelx: %i, pixely: %i, depth: %i, iter: %i", pixelx, pixely, depth, iter);
	//}

//#if 0 == COMPACT
	if (terminatenumber >= path.remainingBounces) { return; }
//#endif

	//Hit Nothing
	if (0.f >= isect.t) {
		path.remainingBounces = terminatenumber;
		return;
	}

	const Material& m = materials[isect.materialId];

	//First Bounce or Last bounce was specular
	//NOTE: may want to remove spec bounce check as the matchesflags from 561 didnt work
	const glm::vec3 wo = -path.ray.direction;
	if (0 == depth || path.specularbounce) { //what if path color started at 1 and you multiplied as you went along???
		path.color += path.throughput * Le(m, isect.surfaceNormal, wo); 
	}

	//Hit Light
	if (0.f < m.emittance) {
		path.remainingBounces = terminatenumber;//already accounted for Le above
		return;
	}

	//if(material doesnt exist) { path.remainingBounces = terminatenumber; return; }

    //pre-check for pure specular, we can skip DL MIS if so.
	path.specularbounce = 0 == m.hasSubSurface && (1 == m.hasReflective || 1 == m.hasRefractive) ? 1 : 0;

	//////////////////////
	//////  DIRECT  //////
	//////////////////////
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);
	const int randlightindex = dev_geomLightIndices[int( u01(rng)*numlights )];
	const Geom& randlight = dev_geoms[randlightindex];
	glm::vec3 pisect = getPointOnRay(path.ray, isect.t) + isect.surfaceNormal*EPSILON;
	glm::vec3 nisect = isect.surfaceNormal;

	glm::vec3 colordirect(0.f);
	MIS_LightSampleContribution(colordirect, isect, pisect, nisect, 
		wo, path, m, randlight, randlightindex, 
		numlights, rng, materials, dev_geoms, numgeoms);

	glm::vec3 colordirectsample(0.f);
	MIS_BxdfSampleContribuition(colordirectsample, colordirect,
		pisect, isect, path, m, randlight, randlightindex, numlights, rng, materials,
		dev_geoms, numgeoms);

	//IF SPECULARBOUNCE, PROB SHOULDNT DO ANY OF THE DIRECT LIGHTING SINCE WE WILL DOUBLE COUNT IT IF IT WAS HEADING TOWARDS A LIGHT ANYWAY
	//taken care of in the direct light importance sampling functions, these components get skipped if specbounce
	path.color += path.throughput*(colordirect + colordirectsample);
    
	//NOTE: uncomment to just get DL and no GL
	//path.remainingBounces = terminatenumber;
	//return;

	////////////////////////
	//////  INDIRECT  //////
	////////////////////////
	//get global illum ray from bxdf and loop
	//do a normal samplef to get wi pdf and color
	float bxdfPDF; glm::vec3 bxdfColor;
	scatterRayNaive(path, isect, m, bxdfPDF, bxdfColor, rng, dev_geoms);
	const glm::vec3 bxdfWi = path.ray.direction;
	const glm::vec3 normal = isect.surfaceNormal;

	glm::vec3 current_throughput(0.f); 
	if(!isBlack(bxdfColor) && 0 < bxdfPDF) { 
		current_throughput = bxdfColor * absDot(bxdfWi, normal) / bxdfPDF;
	}
	path.throughput *= current_throughput;
	
	//russian roulette terminate low-energy rays
	//if(depth > MAXBOUNCES) {//glass can look dimmer if you start terminating earlier (vs the naive)
	if(depth > 3) {
		//in the off chance this ray is still going after a long time
		//scale it up to reduce noise 
		//i.e. make it more like the rays in its pixel who terminated earlier
		//this one just got lucky presumably
		//BOOK:
		//const float q = glm::max(0.05f, 1.f - path.throughput.y);
		//if (q > u01(rng)) { path.remainingBounces = terminatenumber; return; }
		//path.throughput /= (1.f - q);
		const float q = glm::compMax(path.throughput);
		if (q < u01(rng)) { path.remainingBounces = terminatenumber; return; }
		path.throughput /=  q;
	}
}


