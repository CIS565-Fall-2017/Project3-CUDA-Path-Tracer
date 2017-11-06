#pragma once

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "intersections.h"
#include "sampling.h"
#include "materialInteractions.h"
#include "lightInteractions.h"
#include "common.h"

__host__ __device__ void russianRoulette(Color3f& energy, float probability, int& currentdepth, int& recursionLimit)
{
	//set current ray depth to be zero to end iterative call to to fullLighting integrator
	if (currentdepth <= (recursionLimit - 3))
	{
		float highest_energy_component = std::max(std::max(energy.x, energy.y), energy.z);
		energy = energy / highest_energy_component;
		if (highest_energy_component < probability)
		{
			currentdepth = 0;
		}
	}
}

__host__ __device__ float PowerHeuristic(int nf, Float fPdf, int ng, Float gPdf)
{
	float denominator = (fPdf*nf*fPdf*nf + gPdf*ng*gPdf*ng);
	if (fequals(denominator, 0.0f))
	{
		return 0.0f;
	}
	return glm::pow((nf*fPdf), 2) / denominator;
}

__host__ __device__ void naiveIntegrator(PathSegment & pathSegment,
										 ShadeableIntersection& intersection,
										 Material &m, Geom& geom, Geom * geoms, int numGeoms,
										 thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f wo = pathSegment.ray.direction;
	Vector3f wi = glm::vec3(0.0f);
	Vector3f sampledColor = pathSegment.color;
	Vector3f normal = intersection.surfaceNormal;
	BxDFType chosenBxDF;
	float pdf = 0.0f;

	sampledColor = material_sample_f(m, rng, wo, normal, geom, intersection, pathSegment,
									 geoms, numGeoms, sampledColor, wi, pdf, chosenBxDF);

	if (pdf != 0.0f)
	{
		float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
		pathSegment.color *= sampledColor*absdot / pdf;
	}
	
	pathSegment.ray = spawnNewRay(intersection, wi);
	pathSegment.remainingBounces--;
}

__host__ __device__ void directLightingIntegrator(PathSegment & pathSegment,
												  ShadeableIntersection& intersection,
												  Material* materials, Material &m,
												  Geom& geom, Geom* geoms, int numGeoms,
												  Light * lights, int &numLights,
												  thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	thrust::uniform_real_distribution<float> u01(0, 1);
	Vector3f intersectionPoint = intersection.intersectPoint;
	Vector3f normal = intersection.surfaceNormal;

	Color3f finalcolor = Color3f(0.0f);

	Vector3f wo = pathSegment.ray.direction;
	Vector3f wi;
	float pdf;
	Vector2f xi = Vector2f(u01(rng), u01(rng));

	//Assuming the scene has atleast one light
	int randomLightIndex = std::min((int)std::floor(u01(rng)*numLights), numLights - 1);
	Light selectedLight = lights[randomLightIndex];
	Geom lightGeom = geoms[selectedLight.lightGeomIndex];
	Material lightMat = materials[lightGeom.materialid];

	//Sample Light	
	Vector3f wi_towardsLight = glm::vec3(0.0f);
	Color3f sampledLightColor = sampleLights(lightMat, normal, wi_towardsLight, xi, pdf, intersectionPoint, lightGeom);
	pdf /= numLights;

	//sample material of object you hit in the scene
	BxDFType chosenBxDF = chooseBxDF(m, rng);
	Vector3f f = material_f(m, chosenBxDF, wo, wi_towardsLight, rng);

	if (pdf != 0.0f)
	{
		float absdot = glm::abs(glm::dot(wi_towardsLight, intersection.surfaceNormal));
		pathSegment.color = f * sampledLightColor * absdot / pdf;
	}

	//visibility test
	ShadeableIntersection isx;
	Ray shadowFeeler = spawnNewRay(intersection, wi_towardsLight);
	computeIntersectionsForASingleRay(shadowFeeler, isx, geoms, numGeoms);
	
	// if the shadow feeler ray doesnt hit the sample light then color the pathSegment black
	// printf("hitIndex and lightIndex: %d    %d \n", isx.hitGeomIndex, selectedLight.lightGeomIndex);
	if (isx.t > 0.0f && isx.hitGeomIndex != selectedLight.lightGeomIndex)
	{
		pathSegment.color = Color3f(0.f);
	}

	pathSegment.remainingBounces = 0;
}

__host__ __device__ void fullLightingIntegrator(int maxTraceDepth, 
												PathSegment & pathSegment,
												ShadeableIntersection& intersection,
												Material* materials, Material &m, 
												Geom * geoms, Geom& geom, int numGeoms,
												Light * lights, int &numLights,
												Color3f accumulatedThroughput, Color3f accumulatedColor,
												thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f wo = pathSegment.ray.direction;
	Color3f directLightingColor = Color3f(0.0);
	Vector3f normal = intersection.surfaceNormal;
	Vector3f intersectionPoint = intersection.intersectPoint;
	thrust::uniform_real_distribution<float> u01(0, 1);

	//---------------------------------------------------------------------
	//----------------------- Actual Direct Lighting ----------------------
	//---------------------------------------------------------------------
	Vector3f wi_Direct_Light;
	float pdf_Direct_Light;
	Point2f xi_Direct_Light = Point2f(u01(rng), u01(rng));
	Color3f LTE_Direct_Light = Color3f(0.0f);

	int randomLightIndex = -1;
	Material lightMat;
	Geom lightGeom;
	Light selectedLight;
	BxDFType chosenBxDF;

	{
		//Li term - not recursive, cause direct lighting
		randomLightIndex = std::min((int)std::floor(u01(rng)*numLights), numLights - 1);
		selectedLight = lights[randomLightIndex];
		lightGeom = geoms[selectedLight.lightGeomIndex];
		lightMat = materials[lightGeom.materialid];
		Color3f li_Direct_Light = sampleLights(lightMat, normal, wi_Direct_Light, xi_Direct_Light,
												pdf_Direct_Light, intersectionPoint, lightGeom);

		//f term term
		chosenBxDF = chooseBxDF(m, rng);
		Vector3f f_Direct_Light = material_f(m, chosenBxDF, wo, wi_Direct_Light, rng);

		if (pdf_Direct_Light != 0.0f)
		{
			//absDot term		
			float absDot_Direct_Light = glm::abs(glm::dot(wi_Direct_Light, intersection.surfaceNormal));

			//visibility test for direct lighting
			ShadeableIntersection isx;
			Ray n_ray_Direct_Light = spawnNewRay(intersection, wi_Direct_Light);
			computeIntersectionsForASingleRay(n_ray_Direct_Light, isx, geoms, numGeoms);

			// LTE calculation for Direct Lighting
			if (isx.t > 0.0f && isx.hitGeomIndex != selectedLight.lightGeomIndex)
			{
				LTE_Direct_Light = f_Direct_Light * li_Direct_Light * absDot_Direct_Light / pdf_Direct_Light;
			}
		}
	}	

	//---------------------------------------------------------------------
	//----------------------- BSDF based Direct Lighting ------------------
	//---------------------------------------------------------------------
	Vector3f wiW_BSDF_Light;
	float pdf_BSDF_Light;
	Point2f xi_BSDF_Light = Point2f(u01(rng), u01(rng));
	Color3f LTE_BSDF_Light = Color3f(0.0f);
	BxDFType sampledType;
	int randomLightIndex_bsdf = -1;

	{
		//if the object is not pure specular
		if (chosenBxDF != BSDF_SPECULAR_BRDF)
		{
			//f term
			Color3f f_BSDF_Light = material_sample_f(m, rng, wo, normal, geom, intersection, pathSegment,
													geoms, numGeoms, f_BSDF_Light, wiW_BSDF_Light, 
													pdf_BSDF_Light, chosenBxDF);

			if (pdf_BSDF_Light != 0.0f)
			{
				//absDot Term
				float absDot_BSDF_Light = glm::abs(glm::dot(wiW_BSDF_Light, intersection.surfaceNormal));
				
				//visibility test for bsdf based direct lighting; this also tests if the bsdf
				//direct lighting ray actually hit the selected light used for actual direct
				//lighting (Light important sampling based direct lighting)
				ShadeableIntersection isx_bsdf;
				Ray n_ray_BSDF_Light = spawnNewRay(intersection, wiW_BSDF_Light);
				
				computeIntersectionsForASingleRay(n_ray_BSDF_Light, isx_bsdf, geoms, numGeoms);

				// LTE calculation for Direct Lighting
				if (isx_bsdf.t > 0.0f && isx_bsdf.hitGeomIndex != selectedLight.lightGeomIndex)
				{
					//Li term - not recursive, cause direct lighting; emitted Light
					Material m_bsdf_lighthit = materials[isx_bsdf.materialId];
					Color3f li_BSDF_Light = L(m_bsdf_lighthit, isx_bsdf.surfaceNormal, -wiW_BSDF_Light);

					//LTE term
					LTE_BSDF_Light = f_BSDF_Light * li_BSDF_Light * absDot_BSDF_Light / pdf_BSDF_Light;
				}
			}
		}
	}

	//---------------------------------------------------------------------
	//----------------------- MIS -----------------------------------------
	//---------------------------------------------------------------------
	//MIS can only work on one type of light  energy, and so we use MIS for direct lighting only
	float weight_BSDF_Light = 0.0;
	float weight_Direct_Light = 0.0;

	{
		if (randomLightIndex != -1)
		{
			weight_BSDF_Light = PowerHeuristic(1, pdf_BSDF_Light, 1, (sampleLightPDF(lightMat, wiW_BSDF_Light, intersection, lightGeom)));
			weight_Direct_Light = PowerHeuristic(1, pdf_Direct_Light, 1, material_Pdf(m, sampledType, wo, wi_Direct_Light, normal));
		}

		Color3f weighted_BSDF_Light_color = LTE_BSDF_Light * weight_BSDF_Light;
		Color3f weighted_Direct_Light_color = LTE_Direct_Light * weight_Direct_Light;
		directLightingColor = (weighted_BSDF_Light_color + weighted_Direct_Light_color) * (float)numLights;
	}

	//---------------------------------------------------------------------
	//----------------------- Update Overall DirectLightingColor ----------
	//---------------------------------------------------------------------
	//do the check for specular here
	if (pathSegment.remainingBounces == maxTraceDepth)
	{
		directLightingColor += m.emittance * m.color;
	}

	directLightingColor *= pathSegment.accumulatedThroughput;

	//---------------------------------------------------------------------
	//----------------------- Indirect Lighting (Global Illumination) -----
	//---------------------------------------------------------------------
	Vector3f wiW_BSDF_Indirect;
	float pdf_BSDF_Indirect;
	Point2f xi_BSDF_Indirect = Point2f(u01(rng), u01(rng));

	{
		//f term
		Color3f f_BSDF_Indirect = material_sample_f(m, rng, wo, normal, geom, intersection, pathSegment,
													geoms, numGeoms, f_BSDF_Indirect, wiW_BSDF_Indirect,
													pdf_BSDF_Indirect, chosenBxDF);

		if (pdf_BSDF_Indirect != 0.0f)
		{
			//No Li term per se, this is accounted for via accumulatedThroughputColor
			//absDot Term
			float absDot_BSDF_Indirect = glm::abs(glm::dot(wiW_BSDF_Indirect, intersection.surfaceNormal));
			//LTE term
			pathSegment.accumulatedThroughput *= f_BSDF_Indirect * absDot_BSDF_Indirect / pdf_BSDF_Indirect;
		}
	}

	//----------------------- Russian Roulette ----------------------------
	russianRoulette(pathSegment.accumulatedThroughput, u01(rng), pathSegment.remainingBounces, maxTraceDepth);
	//----------------------- Update AccumulatedColor ---------------------
	pathSegment.accumulatedColor += directLightingColor;
	//----------------------- new Ray Generation --------------------------
	pathSegment.ray = spawnNewRay(intersection, wiW_BSDF_Indirect);
	pathSegment.remainingBounces--;
}