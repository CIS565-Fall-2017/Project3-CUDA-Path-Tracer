#pragma once

#include "intersections.h"
#include "sampling.h"
#include "materialInteractions.h"
#include "lightInteractions.h"

__host__ __device__ Ray spawnNewRay(const ShadeableIntersection& intersection, Vector3f& wiW)
{
	Vector3f originOffset = intersection.surfaceNormal * EPSILON;
	// Make sure to flip the direction of the offset so it's in the same general direction as the ray direction
	originOffset = (glm::dot(wiW, intersection.surfaceNormal) > 0) ? originOffset : -originOffset;
	Point3f o(intersection.intersectPoint + originOffset);
	Ray r = Ray();
	r.direction = wiW;
	r.origin = o;
	return r;
}

__host__ __device__ void naiveIntegrator(PathSegment & pathSegment,
										 const ShadeableIntersection& intersection,
										 const Material &m,
										 thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f wo = pathSegment.ray.direction;
	Vector3f wi = glm::vec3(0.0f);
	Vector3f sampledColor = pathSegment.color;
	Vector3f normal = intersection.surfaceNormal;
	float pdf = 0.0f;

	sampleMaterials(m, wo, normal, sampledColor, wi, pdf, rng);

	if (pdf != 0.0f)
	{
		float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
		pathSegment.color *= sampledColor*absdot / pdf;
	}
	
	pathSegment.ray = spawnNewRay(intersection, wi);
	pathSegment.remainingBounces--;
}

__host__ __device__ void directLightingIntegrator(PathSegment & pathSegment,
												const ShadeableIntersection& intersection,
												const Material sceneObjMat, const Material* materials,
												Geom* geoms, const int geom_size, const int num_Rays,
												const int &numLights, const Light * lights,
												thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	Vector3f intersectionPoint = intersection.intersectPoint;
	Vector3f normal = intersection.surfaceNormal;

	Vector3f wo = pathSegment.ray.direction;
	Vector3f wi = glm::vec3(0.0f);
	float pdf = 0.0f;
	Vector2f xi;

	Vector3f sampledLightColor;
	Vector3f f = pathSegment.color;

	//Assuming the scene has atleast one light
	thrust::uniform_real_distribution<float> u01(0, 1);
	
	//int randomLightIndex = glm::min((int)std::floor(u01(rng)*numLights), numLights - 1);
	int randomLightIndex = 1;
	//printf("light index: %d", randomLightIndex);
	Light selectedLight = lights[randomLightIndex];
	Geom lightGeom = geoms[selectedLight.lightGeomIndex];
	Material lightMaterial = materials[lightGeom.materialid];

	//Sample Light
	xi = Vector2f(u01(rng), u01(rng));
	sampledLightColor =  sampleLights(lightMaterial, normal, wi, xi, pdf, intersectionPoint, lightGeom);
	//if (pdf == 0.0f)
	//{
	//	pathSegment.color = Color3f(0.f);
	//	pathSegment.remainingBounces = 0;
	//	return;
	//}
	//else
	//{
	//	pdf /= numLights;
	//}
	

	//calculate f, and only sets pdf to zero if the material has no bsdf
	sampleMaterialsForColor(sceneObjMat, wo, normal, f, wi, pdf, rng);

	if (pdf == 0.0f)
	{
		pathSegment.color = Color3f(0.f);
	}

	//float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
	pathSegment.color = f *sampledLightColor;// / pdf;// *absdot;

	//visibility test
	ShadeableIntersection isx;
	computeIntersectionsWithSelectedObject(pathSegment, geoms, geom_size, isx);
	
	if (isx.t < 0.0f && materials[isx.materialId].emittance <= EPSILON) // if the shadow feeler ray doesnt hit the sample light then color the pathSegment black
	{
		pathSegment.color = Color3f(0.f);
	}

	pathSegment.remainingBounces = 0;
}

/*
Full Lighting

DirectLighting
BSDFBasedDirectLighting
MIS
IndirectBSDFLighting
UpdateRayDirection

//----------------------- Indirect Lighting (Global Illumination) -----
Vector3f wiW_BSDF_Indirect;
float pdf_BSDF_Indirect;
Point2f xi_BSDF_Indirect = sampler->Get2D();

if( depth == recursionLimit  || flag_CameFromSpecular )
{
	directLightingColor += intersection.Le(woW);
}

flag_CameFromSpecular = false;
if(flag_Hit_Specular) //if( (sampledType & BSDF_SPECULAR) == BSDF_SPECULAR )
{
	flag_CameFromSpecular = true;
}

directLightingColor *= accumulatedThroughputColor;  //for only BSDFIndirectLighting, assume directLighting is 1,1,1

if(!flag_NoBSDF)
{
	//f term
	Color3f f_BSDF_Indirect = intersection.bsdf->Sample_f(woW, &wiW_BSDF_Indirect, xi_BSDF_Indirect,
	&pdf_BSDF_Indirect, BSDF_ALL, &sampledType);

	if(pdf_BSDF_Indirect != 0.0f)
	{
		//No Li term per se, this is accounted for via accumulatedThroughputColor

		//absDot Term
		float absDot_BSDF_Indirect = AbsDot(wiW_BSDF_Indirect, intersection.normalGeometric);

		//LTE term
		accumulatedThroughputColor *= f_BSDF_Indirect * absDot_BSDF_Indirect / pdf_BSDF_Indirect;
	}
}

flag_Terminate = RussianRoulette(accumulatedThroughputColor, probability, depth); //can change accumulatedThroughput
accumulatedRayColor += directLightingColor;

Ray n_ray_BSDF_Indirect = intersection.SpawnRay(wiW_BSDF_Indirect);
woW = -n_ray_BSDF_Indirect.direction;
r = n_ray_BSDF_Indirect;
depth--;
}

return accumulatedRayColor;

*/