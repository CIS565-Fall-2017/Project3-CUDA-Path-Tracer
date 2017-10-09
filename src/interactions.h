#pragma once

#include "intersections.h"




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


// Return hit_geom_index
__host__ __device__ int intersectScene(
	  const Ray& shadowFeelRay
	, Geom * geoms
	, Triangle * tris
	, int geomSize
#ifdef ENABLE_MESHWORLDBOUND
	, Bounds3f * worldBounds
#endif
#ifdef ENABLE_BVH
	, LinearBVHNode * nodes
#endif
) 
{
	float t = 0.f;
	float t_min = FLT_MAX;
	glm::vec2 tmp_uv;
	glm::vec3 tmp_normal;
	glm::mat3 tmp_tangentToWorld;
	int hit_geom_index = -1;
	bool outside;

	for (int i = 0; i < geomSize; i++)
	{
		Geom & geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, shadowFeelRay, tmp_uv, tmp_normal, outside, tmp_tangentToWorld);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, shadowFeelRay, tmp_uv, tmp_normal, outside, tmp_tangentToWorld);
		}
		// TODO: add more intersection tests here... triangle? metaball? CSG?
		else if (geom.type == MESH)
		{
#ifdef ENABLE_MESHWORLDBOUND
#ifdef ENABLE_BVH
			ShadeableIntersection temp_isect;
			temp_isect.t = FLT_MAX;
			int hit_tri_index = -1;

			if (IntersectBVH(shadowFeelRay, &temp_isect, hit_tri_index, nodes, tris)) {
				if (hit_tri_index >= geom.meshTriangleStartIdx
				 && hit_tri_index < geom.meshTriangleEndIdx) {
					t = temp_isect.t;
				}
				else {
					t = -1.0f;
				}
			}
			else {
				t = -1.0f;
			}
#else
			// Check geom related world bound first
			float tmp_t;
			if (worldBounds[geom.worldBoundIdx].Intersect(shadowFeelRay, &tmp_t)) {
				t = meshIntersectionTest(geom, tris, shadowFeelRay, tmp_uv, tmp_normal, outside);
			}
			else {
				t = -1.0f;
			}
#endif
#else
			// loop through all triangles in related mesh
			t = meshIntersectionTest(geom, tris, shadowFeelRay, tmp_uv, tmp_normal, outside);
#endif		
		}
		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
		}
	}

	return hit_geom_index;
}

// For glass material
__host__ __device__
inline float FrDielectric(float cosThetaI, float etaI, float etaT) {

	cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

	// Potentially swap indices of refraction
	bool entering = cosThetaI > 0.f;
	if (!entering) {
		float tmp = etaI;
		etaI = etaT;
		etaT = tmp;
		cosThetaI = glm::abs(cosThetaI);
	}

	// Compute cosThetaT using Snell's law
	float sinThetaI = glm::sqrt(glm::max(0.f, 1.0f - cosThetaI * cosThetaI));

	float sinThetaT = etaI / etaT * sinThetaI;

	// Handle total internal reflection
	if (sinThetaT >= 1.0f) {
		return 1.0f;
	}

	float cosThetaT = glm::sqrt(glm::max(0.f, 1.0f - sinThetaT * sinThetaT));


	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
		((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rprep = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
		((etaI * cosThetaI) + (etaT * cosThetaT));

	return (Rparl * Rparl + Rprep * Rprep) / 2.0f;
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
void scatterRay(
		PathSegment& pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
		glm::vec3 normal_bsdf,
		glm::vec2 uv,
        const Material &m,
        thrust::default_random_engine &rng,
		Texture * texutres,
		Texture * environmentMap
#ifdef	ENABLE_DIR_LIGHTING
		,const Light* lights
		,const int& lightSize
		, Geom * geoms
		, Triangle * tris
		, int geomSize
#ifdef ENABLE_MESHWORLDBOUND
		, Bounds3f * worldBounds
#endif
#ifdef ENABLE_BVH
		, LinearBVHNode * nodes
#endif
#endif
) {

	// get probabilty
	thrust::uniform_real_distribution<float> u01(0, 1);
	float probability = u01(rng);
	glm::vec3 incidentDirection = pathSegment.ray.direction;
	glm::vec3 newDirection;


	// ------------------- Glass : Specular reflection & refraction-------------------
	bool isGlass = (m.hasReflective != 0.f && m.hasRefractive != 0.f);
	if (isGlass) {
		float specularSum = m.hasReflective + m.hasRefractive;
		float reflecProb = m.hasReflective / specularSum;
		float refractProb = 1.0f - reflecProb;

		// Use Schlick's Approximation. No specific Frensel model
	//	float temp = (1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction);
	//	float R0 = temp * temp;
	//	float OneMinusCosineTheta = 1.0f - AbsDot(incidentDirection, normal_bsdf);
	//	float frenselCoefficient = R0 + (1.0f - R0) * OneMinusCosineTheta * OneMinusCosineTheta * OneMinusCosineTheta * OneMinusCosineTheta * OneMinusCosineTheta;

		float frenselCoefficient;

		if (probability < reflecProb) {
			newDirection = Reflect(-incidentDirection, normal_bsdf);
			newDirection = glm::normalize(newDirection);
			pathSegment.ray.direction = newDirection;
			pathSegment.ray.origin = intersect + 0.0001f * normal_bsdf;

			frenselCoefficient = FrDielectric( glm::dot(normal_bsdf,newDirection), 1.0f, m.indexOfRefraction);

#ifndef ENABLE_DIR_LIGHTING
			pathSegment.color *= (frenselCoefficient * m.specular.color / reflecProb); // divide the probability to counter the chance
#else
#ifdef ENABLE_MIS_LIGHTING
			pathSegment.ThroughputColor *= (frenselCoefficient * m.specular.color / reflecProb); // divide the probability to counter the chance
#endif
#endif
		}

		// if it's specualr and not reflective
		// then it's Refractive
		else
		{
			bool entering = glm::dot(incidentDirection, normal_bsdf) < 0;
			float indexOfRefraction = m.indexOfRefraction;
			float eta = entering ? (1.0f / indexOfRefraction) : indexOfRefraction;

			glm::vec3 wt;
			if (!Refract(-incidentDirection, Faceforward(normal_bsdf, -incidentDirection), eta, &wt)) {
				newDirection = Reflect(-incidentDirection, Faceforward(normal_bsdf, -incidentDirection));
				newDirection = glm::normalize(newDirection);

				frenselCoefficient = FrDielectric(glm::dot(normal_bsdf, newDirection), 1.0f, m.indexOfRefraction);

				pathSegment.ray.direction = newDirection;
				pathSegment.ray.origin = intersect + 0.0001f * normal_bsdf;
			}
			else {
				newDirection = glm::normalize(wt);

				frenselCoefficient = FrDielectric(glm::dot(normal_bsdf, newDirection), 1.0f, m.indexOfRefraction);
				
				pathSegment.ray.direction = newDirection;
				pathSegment.ray.origin = intersect + 0.0002f * incidentDirection + 0.0002f * Faceforward(normal_bsdf, incidentDirection);
			}
#ifndef ENABLE_DIR_LIGHTING
			pathSegment.color *= ((1.f - frenselCoefficient) * m.specular.color / refractProb);  // divide the probability to counter the chance
#else
#ifdef ENABLE_MIS_LIGHTING
			pathSegment.ThroughputColor *= ((1.f - frenselCoefficient) * m.specular.color / refractProb);  // divide the probability to counter the chance
#endif
#endif
		}
	}

	// ----------------- Pure Specular reflection -------------------------
	//else if (m.hasReflective == 1.f && m.hasRefractive == 0.f) {
	else if (probability < m.hasReflective) {
		newDirection = Reflect(-incidentDirection, normal_bsdf);
		newDirection = glm::normalize(newDirection);
		pathSegment.ray.direction = newDirection;
		pathSegment.ray.origin = intersect + 0.0001f * normal_bsdf;

#ifndef ENABLE_DIR_LIGHTING
		pathSegment.color *= m.specular.color; 
#else
#ifdef ENABLE_MIS_LIGHTING
		pathSegment.ThroughputColor *= m.specular.color;
#endif
#endif
	}

	// ----------------- Pure Specular refraction ----------------------------
	//else if (m.hasReflective == 0.f && m.hasRefractive == 1.f) {
	else if (probability < m.hasRefractive) {

		bool entering = glm::dot(incidentDirection, normal_bsdf) < 0;
		float indexOfRefraction = m.indexOfRefraction;
		float eta = entering ? (1.0f / indexOfRefraction) : indexOfRefraction;

		//glm::vec3 newDirection = glm::refract(incidentDirection, normal, eta);
		//pathSegment.ray.direction = newDirection;
		//pathSegment.ray.origin = intersect + 0.0002f * newDirection;
		//pathSegment.color *= m.specular.color;

		glm::vec3 wt;
		if (!Refract(-incidentDirection, Faceforward(normal_bsdf, -incidentDirection), eta, &wt)) {
			newDirection = Reflect(-incidentDirection, Faceforward(normal_bsdf, -incidentDirection));
			newDirection = glm::normalize(newDirection);
			pathSegment.ray.direction = newDirection;
			pathSegment.ray.origin = intersect + 0.0001f * normal_bsdf;
		}
		else {
			pathSegment.ray.direction = glm::normalize(wt);
			pathSegment.ray.origin = intersect + 0.0002f * incidentDirection + 0.0002f * Faceforward(normal_bsdf, incidentDirection);
		}

#ifndef ENABLE_DIR_LIGHTING
		pathSegment.color *= m.specular.color;  
#else
#ifdef ENABLE_MIS_LIGHTING
		pathSegment.ThroughputColor *= m.specular.color;
#endif
#endif
	}

	// ------------------- Non-specular / Diffuse Part ---------------------
	else 
	{
		glm::vec3 real_color = m.color;
		if (m.textureID != -1) {
			// get texture color here
			real_color *= texutres[m.textureID].getColor(uv);
		}


#ifdef ENABLE_DIR_LIGHTING
		if (pathSegment.remainingBounces == 0) {
			return;
		}
		
		probability = u01(rng);

		if (lightSize == 0) { pathSegment.color = glm::vec3(0.f); return; }


		// randomly select a light
		int selectLightIdx = glm::min((int)(probability * lightSize), lightSize - 1);


		// Sample a point on the light and get solid-angle related pdf
		
		float pdf_direct = 0.f;
		const Light& selectedLight = lights[selectLightIdx];
		glm::vec3 pointOnLight;


		if (selectedLight.type == DiffuseAreaLight) {
			float random_x = u01(rng);
			float random_y = u01(rng);

			pointOnLight = selectedLight.sample(random_x, random_y, &pdf_direct);

			float distanceSquared = glm::distance2(intersect, pointOnLight);
			pdf_direct *= distanceSquared / AbsDot(normal, glm::normalize(pointOnLight - intersect));
		}

#ifndef ENABLE_MIS_LIGHTING
		pdf_direct /= (float)lightSize;
#endif

		Ray shadowFeelRay;
		shadowFeelRay.origin = intersect;


		if (selectedLight.type == DiffuseAreaLight) {
			shadowFeelRay.direction = glm::normalize(pointOnLight - intersect);
		}
		else if (selectedLight.type == InfiniteAreaLight) {
			// if it's an environment light map, use a cosine-weight distribution
			shadowFeelRay.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal_bsdf, rng));
			pdf_direct = AbsDot(shadowFeelRay.direction, normal_bsdf) * InvPi; // Trick.. For simplicity, just use Lambert bxdf pdf here
		}



#ifndef ENABLE_MIS_LIGHTING
		if (intersectScene(shadowFeelRay, geoms, tris, geomSize
#ifdef ENABLE_MESHWORLDBOUND
						   ,worldBounds
#endif
#ifdef ENABLE_BVH
						   ,nodes
#endif
							) == selectedLight.geomIdx) {

			pathSegment.color += (((real_color / PI) * selectedLight.emittance * AbsDot(normal, shadowFeelRay.direction) / pdf_direct) / 2.f);
		}
#else

		// ******************* MIS  START *************************
		glm::vec3 throughputColor = pathSegment.ThroughputColor;

		// ********** MIS Direct lighting part ********************
		glm::vec3 OneLightColor(0.f);

		float pdf_bsdf = 1.0f / PI; // Diffuse bxdf pdf value

		// use power heuristic here 
		float weigh_direct = (pdf_direct * pdf_direct) / (pdf_direct * pdf_direct + pdf_bsdf * pdf_bsdf);
		float weigh_bsdf = 1.0f - weigh_direct;

		if (intersectScene(shadowFeelRay, geoms, tris, geomSize
#ifdef ENABLE_MESHWORLDBOUND
						   , worldBounds
#endif
#ifdef ENABLE_BVH
						   , nodes
#endif
						   ) == selectedLight.geomIdx) {
			if (selectedLight.type == DiffuseAreaLight) {
				OneLightColor += ((real_color / PI) * selectedLight.emittance * AbsDot(normal, shadowFeelRay.direction) * weigh_direct / pdf_direct);
			}
			else if (selectedLight.type == InfiniteAreaLight) {
				OneLightColor += ((real_color / PI) * environmentMap->getEnvironmentColor(shadowFeelRay.direction) * AbsDot(normal, shadowFeelRay.direction) * weigh_direct / pdf_direct);
			}
		}

		// ********** MIS Direct bsdf part ********************
		newDirection = calculateRandomDirectionInHemisphere(normal_bsdf, rng);
		//shadowFeelRay.origin = intersect;
		shadowFeelRay.direction = glm::normalize(newDirection);

		if (intersectScene(shadowFeelRay, geoms, tris, geomSize
#ifdef ENABLE_MESHWORLDBOUND
						   , worldBounds
#endif
#ifdef ENABLE_BVH
						   , nodes
#endif
						  ) == selectedLight.geomIdx) {
			if (selectedLight.type == DiffuseAreaLight) {
				OneLightColor += ((real_color / PI) * selectedLight.emittance * AbsDot(normal, shadowFeelRay.direction) * weigh_bsdf / pdf_bsdf);
			}
			else if (selectedLight.type == InfiniteAreaLight) {
				OneLightColor += ((real_color / PI) * environmentMap->getEnvironmentColor(shadowFeelRay.direction) * AbsDot(normal, shadowFeelRay.direction) * weigh_bsdf / pdf_bsdf);
			}
		}

		/*throughputColor *= ((float)lightSize * OneLightColor);
		pathSegment.ThroughputColor = throughputColor;*/

		pathSegment.color += throughputColor * ((float)lightSize * OneLightColor);


		// ************** MIS set new ray ******************
		newDirection = calculateRandomDirectionInHemisphere(normal_bsdf, rng);
		pathSegment.ray.direction = glm::normalize(newDirection);
		pathSegment.ray.origin = intersect;
		pathSegment.ThroughputColor = throughputColor * real_color;

		//pathSegment.ThroughputColor = throughputColor * real_color * AbsDot(pathSegment.ray.direction, normal) ;

		// ************** MIS  END *************************


#endif



#else
		newDirection = calculateRandomDirectionInHemisphere(normal_bsdf, rng);
		newDirection = glm::normalize(newDirection);
		pathSegment.ray.direction = newDirection;
		pathSegment.ray.origin = intersect;

		// Debug Normal
		//pathSegment.color *= normal;



		// calculateRandomDirectionInHemisphere generates cosine-weight rays
		// -> not multiply AbsDot(wi, N) is OK (probe itself is what we want)
		// Lambert pdf == 1 / Pi
		// Lambert f == Albedo / Pi 
		// -> f * AbsDot / pdf == Albedo(Color) So we directly multiply color here
		pathSegment.color *= real_color;
#endif
	}
}
