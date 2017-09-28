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

__global__ void checkTerminationConditions(int num_paths, ShadeableIntersection * shadeableIntersections,
											PathSegment * pathSegments, Material * materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces <= 0)
		{
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		Material material = materials[intersection.materialId];

		//If the ray didnt intersect with objects in the scene
		if (intersection.t < 0.0f)
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0; //to make thread stop executing things
			return;
		}

		//If the ray hit a light in the scene
		if (material.emittance > 0.0f)
		{
			pathSegments[idx].color *= material.color*material.emittance;
			pathSegments[idx].remainingBounces = 0; //equivalent of breaking out of the thread
			return;
		}
	}
}

__host__ __device__ void naiveIntegrator(PathSegment & pathSegment,
										 ShadeableIntersection& intersection,
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

__host__ __device__ Geom GetLightGeom( const Geom* geoms, const int num_paths,
									   const int &numLights, const Light * lights,
									   const thrust::default_random_engine &rng )
{
	//assuming the scene has atleast one light
	thrust::uniform_real_distribution<float> u01(0, 1);
	float randnum = u01(rng);
	int randomlightindex = std::min((int)std::floor(randnum*numLights), numLights - 1);
	Light selectedlight = lights[randomlightindex];
	return geoms[selectedlight.lightGeomIndex];
}

__host__ __device__ void directLightingIntegrator(PathSegment & pathSegment,
												  ShadeableIntersection& intersection,
												  Geom* geoms, int num_paths,
												  const int &numLights, const Light * lights,
												  const Material &m, thrust::default_random_engine &rng)
{
	// Update the ray and color associated with the pathSegment
	//Vector3f wo = pathSegment.ray.direction;
	//Vector3f wi;
	//Vector3f sampledLightColor;
	//Vector3f intersectionPoint = intersection.intersectPoint;
	//Vector3f normal = intersection.surfaceNormal;
	//Vector2f xi;
	//float pdf = 0.0f;

	////Assuming the scene has atleast one light
	//thrust::uniform_real_distribution<float> u01(0, 1);
	//float randNum = u01(rng);
	//int randomLightIndex = std::min((int)std::floor(randNum*numLights), numLights - 1);
	//Light selectedLight = lights[randomLightIndex];
	//Geom lightGeom = geoms[selectedLight.lightGeomIndex];

	////Sample Light
	//xi = Vector2f(u01(rng), u01(rng));
	//sampledLightColor = sampleLights(m, normal, wi, xi, pdf, intersectionPoint, lightGeom);
	//pdf /= numLights;

	//if (pdf == 0.0f)
	//{
	//	pathSegment.color = Color3f(0.f);
	//}

	//Color3f f;
	//sampleMaterials(m, wo, normal, f, wi, pdf, rng);

	//float absdot = glm::abs(glm::dot(wi, intersection.surfaceNormal));
	//pathSegment.ray = spawnNewRay(intersection, wi);
	//
	////visibility test
	//ShadeableIntersection isx;
	//computeIntersectionsWithSelectedObject(pathSegment, lightGeom, isx);
	//
	//if (isx.t < 0.0f) // if the shadow feeler ray doesnt hit the sample light then color the pathSegment black
	//{
	//	pathSegment.color = Color3f(0.f);
	//}

	//pathSegment.color = (f * sampledLightColor * absdot) / pdf;	
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