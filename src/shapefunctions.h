#pragma once
#include "glm/gtx/norm.hpp"
#include "warpfunctions.h"
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

////////////////////////////////
////////// GET AREAS////////////
////////////////////////////////

__host__ __device__ float getAreaSphere(const Geom& g) {
	return 4.f * PI * g.scale.x * g.scale.x; // We're assuming uniform scale
}

__host__ __device__ float getAreaPlane(const Geom& g) {
	return g.scale.x * g.scale.y; 
}

__host__ __device__ float getAreaCube(const Geom& g) {
	const float twoXY = 2* g.scale.x * g.scale.y; 
	const float twoXZ = 2* g.scale.x * g.scale.z; 
	const float twoYZ = 2* g.scale.y * g.scale.z; 
	return twoXY + twoXZ + twoYZ;
}

__host__ __device__ float getAreaShape(const Geom& g) {
	if (g.type == GeomType::CUBE) {
		return getAreaCube(g);
	} else if (g.type == GeomType::SPHERE) {
		return getAreaSphere(g);
	} else if (g.type == GeomType::PLANE) {
		return getAreaPlane(g);
	}
}

//////////////////////////////////////////
//////////SPHERE SURFACE SAMPLING/////////
//////////////////////////////////////////
__host__ __device__ glm::vec3 surfaceSampleSphereAny(glm::vec3& nlightsamp, const Geom& randlight,
	thrust::default_random_engine& rng, float& pdf) 		
{

	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 pObj = WarpFunctions::squareToSphereUniform( glm::vec2(u01(rng), u01(rng)) );

    nlightsamp = glm::normalize( randlight.invTranspose * pObj );
    glm::vec3 plightsamp = glm::vec3(randlight.transform * glm::vec4(pObj.x, pObj.y, pObj.z, 1.0f));

    pdf = 1.f / getAreaSphere(randlight);

	return plightsamp;
	
}

__host__ __device__ glm::vec3 surfaceSampleSphere(glm::vec3& nlightsamp, const Geom& randlight,
	const glm::vec3& pisect, const glm::vec3& nisect,
	thrust::default_random_engine& rng, float& pdf) {
	glm::vec3 center = randlight.translation;
	glm::vec3 centerToRef = glm::normalize(center - pisect);
	glm::vec3 tan, bit;

	CoordinateSystem(centerToRef, tan, bit);

	glm::vec3 pOrigin;
	if (glm::dot(center - pisect, nisect) > 0) {
		pOrigin = pisect + nisect * EPSILON;//i belived pisect is already offset along the normal before it gets here
	} else {
		pOrigin = pisect - nisect * EPSILON;
	}

	//r=1 in obj space, inside sphere test
	if (glm::distance2(pOrigin, center) <= 1.f) {// r^2 also 1
		return surfaceSampleSphereAny(nlightsamp, randlight, rng, pdf);
	}

	thrust::uniform_real_distribution<float> u01(0, 1);
	const float xi_x = u01(rng);
	const float xi_y = u01(rng);
    float sinThetaMax2 = 1.f / glm::distance2(pisect, center); // r is 1
    float cosThetaMax = std::sqrt(std::fmax(0.f, 1.f - sinThetaMax2));
    float cosTheta = (1.f - xi_x) + xi_x * cosThetaMax;
    float sinTheta = std::sqrt(glm::max(0.f, 1.f - cosTheta * cosTheta));
    float phi = xi_y * 2.f * PI;

    float dc = glm::distance(pisect, center);
    float ds = dc * cosTheta - glm::sqrt(glm::max(0.f, 1.f - dc * dc * sinTheta * sinTheta));

    float cosAlpha = (dc * dc + 1.f - ds * ds) / (2.f * dc * 1.f);
    float sinAlpha = glm::sqrt(glm::max(0.f, 1.f - cosAlpha * cosAlpha));

    glm::vec3 nObj = sinAlpha * glm::cos(phi) * -tan + sinAlpha * glm::sin(phi) * -bit + cosAlpha * -centerToRef;
    glm::vec3 pObj = nObj; // mult by r, but r=1 in obj space

// pObj is in obj space with r=1, so no need to normalize and scale
//    pObj *= radius / glm::length(pObj); 

    glm::vec3 plightsamp = glm::vec3(randlight.transform * glm::vec4(pObj.x, pObj.y, pObj.z, 1.f));
    nlightsamp = randlight.invTranspose * nObj;

    pdf = 1.f / (2.f * PI * (1.f - cosThetaMax));
    return plightsamp;
}

__host__ __device__ glm::vec3 surfaceSamplePlane(glm::vec3& nlightsamp, const Geom& randlight,
	const glm::vec3& pisect, const glm::vec3& nisect,
	thrust::default_random_engine& rng, float& pdf) 
{
    pdf = 1.f / getAreaPlane(randlight);
    nlightsamp = glm::normalize(randlight.invTranspose *glm::vec3(0.f,0.f,1.f));
	thrust::uniform_real_distribution<float> u01(0, 1);
	const float xi_x = u01(rng);
	const float xi_y = u01(rng);
    glm::vec3 plightsamp = glm::vec3(randlight.transform * glm::vec4(xi_x - 0.5f, xi_y - 0.5f, 0.f, 1.f));
	return plightsamp;
}

/////////////////////////////////////
//////// PLANAR SHAPE SAMPLING///////
/////////////////////////////////////
__host__ __device__ glm::vec3 surfaceSampleShape(glm::vec3& nlightsamp, const Geom& randlight,
	const glm::vec3& pisect, const glm::vec3& nisect,
	thrust::default_random_engine& rng, float& pdf) 
{
	//Sphere does not need the view cone estimation like planar shapes do
	if (randlight.type == GeomType::SPHERE) {
		return surfaceSampleSphere(nlightsamp, randlight, pisect, nisect, rng, pdf);

	} else {
		glm::vec3 plightsamp(0.f);
		if (randlight.type == GeomType::CUBE) {
			//TODO: give CUBE its own surface sampling, otherwise the cube light can't be near anything
			//screws up occlusion check for really close objects to the light
			//set it to plightsamp, dont return yet, like the other planar shapes
			return surfaceSampleSphere(nlightsamp, randlight, pisect, nisect, rng, pdf);

		} else if (randlight.type == GeomType::PLANE) {
			plightsamp = surfaceSamplePlane(nlightsamp, randlight, pisect, nisect, rng, pdf);
		}

		{///PDF calc for projected view cone in hemi for planar shapes
			const glm::vec3 wi = glm::normalize(plightsamp - pisect);
			const float coslight = absDot(nlightsamp, -wi);
			const float denom = coslight*(1.f / pdf);
			if (denom != 0.f) {
				pdf = glm::distance2(pisect, plightsamp) / denom;
			} else {
				pdf = 0.f;
			}
		}
		return plightsamp;
	}
}