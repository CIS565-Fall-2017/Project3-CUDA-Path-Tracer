#pragma once

#include <string>
#include <vector>
#include <memory>
#include "glm\gtx\intersect.hpp"
#include "bounds.h"

#include <iostream>
#include <stb_image.h>
#include <cmath>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define COLORDIVIDOR 0.003921568627f


#define InvPi 0.31830988618379067154f
#define Inv2Pi 0.15915494309189533577f

#define Pi 3.14159265358979323846f

// ----------------------------------------------------------------
//----------------------- Toggle Here -----------------------------
// ----------------------------------------------------------------

// Uncomment to enable direct lighting
#define ENABLE_DIR_LIGHTING


// Should define dir lighting first to define mis
#define ENABLE_MIS_LIGHTING


// ----------------------------------------------------------------
// ----------------------------------------------------------------

enum GeomType {
    SPHERE,
    CUBE,
	MESH
};

struct Geom {
    enum GeomType type;
    int materialid;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

	//Only used in mesh cases
	int meshTriangleStartIdx;
	int meshTriangleEndIdx;
	int worldBoundIdx;
};


enum LightType {
	DiffuseAreaLight,
	InfiniteAreaLight
};

struct Light
{	
	enum LightType type;
	Geom geom; // not used in Environment light case
	int geomIdx; // -1 in Environment light case
	float SurfaceArea; // not used in Environment light case
	//float selectedProb; // based on surface area
	glm::vec3 emittance; // not used in Environment light case, sample on environment light map, instead.

	__host__ __device__ glm::vec3 sample(const float& x, const float& y, float* pdf) const {

		if (type == DiffuseAreaLight) {
			glm::vec3 pOnObj(0.f);
			if (geom.type == CUBE) {
				/*float sum = x + y;
				if (sum < 0.33f) {
					pOnObj = glm::vec3(x - 0.5f, y - 0.5f, 0.5001f);
				}
				else if (sum < 0.67f) {
					pOnObj = glm::vec3(x - 0.5f, y - 0.5f, -0.5001f);

				}
				else if (sum < 1.0f) {
					pOnObj = glm::vec3(x - 0.5f, 0.5001f, y - 0.5f);
				}
				else if (sum < 1.33f) {
					pOnObj = glm::vec3(x - 0.5f, -0.5001f, y - 0.5f);
				}
				else if (sum < 1.67f) {
					pOnObj = glm::vec3(0.5001f, x - 0.5f, y - 0.5f);
				}
				else if (sum < 2.0f) {
					pOnObj = glm::vec3(-0.5001f, x - 0.5f, y - 0.5f);
				}*/


				// Simply regard as plane here
				pOnObj = glm::vec3(x - 0.5f, -0.5f, y - 0.5f);
			}
			// Add more geom type sample methods here
			// ......

			//(*pdf) = 1.0f / SurfaceArea;
			(*pdf) = 1.0f / (geom.scale.x * geom.scale.z);
			return glm::vec3(geom.transform * glm::vec4(pOnObj, 1.0f));
		}

		else if (type == InfiniteAreaLight) {
			// Trick here!
			// See scatterRay()

			//float theta = y * Pi, phi = x * 2 * Pi;
			//float cosTheta = glm::cos(theta), sinTheta = glm::sin(theta);
			//float sinPhi = glm::sin(phi), cosPhi = glm::cos(phi);
			//glm::vec3 sample_dir = glm::vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);


			//// Compute PDF for sampled infinite light direction
			//(*pdf) = 1.0f / (4.0f * Pi * 20.0f * 20.0f);

			//return sample_dir;
		}
	}
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
	int textureID;
	int normalID;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
#ifdef ENABLE_MIS_LIGHTING
	glm::vec3 ThroughputColor;
#endif
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  int materialId;

  glm::mat3 tangentToWorld;

  // Sort our pathSegments and related ShadeableIntersections by materialTypeID
  // Used in thrust::sort_by_key
  bool operator<(const ShadeableIntersection& that) const { return this->materialId < that.materialId; }
  bool operator>(const ShadeableIntersection& that) const { return this->materialId > that.materialId; }
};


struct RadixSortElement {
	int OriIndex;
	int materialId;
};

__host__ __device__ inline void CoordinateSystem(glm::vec3& v1, glm::vec3* v2, glm::vec3* v3)
{
	if (std::abs(v1.x) > std::abs(v1.y))
		*v2 = glm::vec3(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
	else
		*v2 = glm::vec3(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
	*v3 = glm::cross(v1, *v2);
}

struct Triangle {
	int index;
	glm::vec3 vertices[3];
	glm::vec3 normals[3];
	glm::vec2 uvs[3];

	__host__ __device__ Bounds3f WorldBound(){
		float minX = glm::min(vertices[0].x, glm::min(vertices[1].x, vertices[2].x));
		float minY = glm::min(vertices[0].y, glm::min(vertices[1].y, vertices[2].y));
		float minZ = glm::min(vertices[0].z, glm::min(vertices[1].z, vertices[2].z));

		float maxX = glm::max(vertices[0].x, glm::max(vertices[1].x, vertices[2].x));
		float maxY = glm::max(vertices[0].y, glm::max(vertices[1].y, vertices[2].y));
		float maxZ = glm::max(vertices[0].z, glm::max(vertices[1].z, vertices[2].z));


		if (minX == maxX) {
			minX -= 0.01f;
			maxX += 0.01f;
		}

		if (minY == maxY) {
			minY -= 0.01f;
			maxY += 0.01f;
		}

		if (minZ == maxZ) {
			minZ -= 0.01f;
			maxZ += 0.01f;
		}


		return Bounds3f(glm::vec3(minX, minY, minZ),
			glm::vec3(maxX, maxY, maxZ));
	}
	__host__ __device__ float SurfaceArea() {
		return glm::length(glm::cross(vertices[0] - vertices[1], vertices[2] - vertices[1])) * 0.5f;
	}

	__host__ __device__ bool Intersect(const Ray& r, ShadeableIntersection* isect) const{

		glm::vec3 baryPosition(0.f);

		if (glm::intersectRayTriangle(r.origin, r.direction,
									  vertices[0], vertices[1], vertices[2],
									  baryPosition)) 
		{

			// Material ID should be set on the Geom level

			isect->t = baryPosition.z;
			/*isect->uv = uvs[0] * baryPosition.x +
						uvs[1] * baryPosition.y +
					    uvs[2] * (1.0f - baryPosition.x - baryPosition.y);*/

			isect->uv = uvs[0] * (1.0f - baryPosition.x - baryPosition.y) +
						uvs[1] * baryPosition.x +
						uvs[2] * baryPosition.y;

			/*isect->surfaceNormal= normals[0] * baryPosition.x +
								  normals[1] * baryPosition.y +
								  normals[2] * (1.0f - baryPosition.x - baryPosition.y);*/

			isect->surfaceNormal = normals[0] * (1.0f - baryPosition.x - baryPosition.y) +
								   normals[1] * baryPosition.x +
								   normals[2] * baryPosition.y;


			glm::vec3 objspaceIntersection = r.origin + (baryPosition.z - .0001f) * glm::normalize(r.direction);

			glm::vec3 pos1 = vertices[1];
			glm::vec3 pos2 = vertices[2];

			glm::vec2 uv1 = uvs[1];
			glm::vec2 uv2 = uvs[2];


			// Edges of the triangle : postion delta
			glm::vec3 deltaPos1 = pos1 - objspaceIntersection;
			glm::vec3 deltaPos2 = pos2 - objspaceIntersection;;

			// UV delta
			glm::vec2 deltaUV1 = uv1 - isect->uv;
			glm::vec2 deltaUV2 = uv2 - isect->uv;


			float r_temp = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
			glm::vec3 tangent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y)*r_temp;
			glm::vec3 bitangent = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x)*r_temp;

			tangent = glm::normalize(tangent);
			bitangent = glm::normalize(bitangent);

			/*if (glm::isnan(tangent.x) ||
				glm::isnan(tangent.y) ||
				glm::isnan(tangent.z)) {
				CoordinateSystem(isect->surfaceNormal, &tangent, &bitangent);
			}
			if (glm::isnan((isect->surfaceNormal).x) ||
				glm::isnan((isect->surfaceNormal).y) ||
				glm::isnan((isect->surfaceNormal).z)) {
				isect->surfaceNormal = glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
				CoordinateSystem(isect->surfaceNormal, &tangent, &bitangent);
			}
			if (glm::isnan(bitangent.x) ||
				glm::isnan(bitangent.y) ||
				glm::isnan(bitangent.z)) {
				CoordinateSystem(isect->surfaceNormal, &tangent, &bitangent);
			}*/

			isect->tangentToWorld = glm::mat3(tangent, bitangent, isect->surfaceNormal);

			return true;
		}

		else
		{
			isect->t = -1.0f;
			return false;
		}

	}
};

struct Texture {
	int width;
	int height;
	int n_comp;
	unsigned char *host_data;
	unsigned char *dev_data;

	void LoadFromFile(char const *filename) {
		host_data = stbi_load(filename, &width, &height, &n_comp, 0);
		if (host_data == NULL) {
				std::cout << "ERROR : texture load fail!" << std::endl;
		}
		dev_data = NULL;
	}

	void Free() {
		// Already done CPU Side Free on pathtraceInit()
		stbi_image_free(host_data);
	}

	// ONLY Used on Device side
	__host__ __device__
	glm::vec3 getColor(glm::vec2& uv) {
		int X = glm::min((float)width * uv.x, (float)width - 1.0f);
		int Y = glm::min((float)height * (1.0f - uv.y), (float)height - 1.0f);

		int texel_index = Y * width + X;

		//Assume n_comp is always 3, which includes red, green and blue three channels
		glm::vec3 col = glm::vec3(dev_data[texel_index * n_comp], dev_data[texel_index * n_comp + 1], dev_data[texel_index * n_comp + 2]);
		col = COLORDIVIDOR * col;
		return col;
	}


	// ONLY Used on Device side
	__host__ __device__
	glm::vec3 getNormal(glm::vec2& uv) {
		int X = glm::min((float)width * uv.x, (float)width - 1.0f);
		int Y = glm::min((float)height * (1.0f - uv.y), (float)height - 1.0f);
		int texel_index = Y * width + X;

		glm::vec3 normal = glm::vec3(dev_data[texel_index * n_comp], dev_data[texel_index * n_comp + 1], dev_data[texel_index * n_comp + 2]);
		normal = 2.0f * COLORDIVIDOR * normal;
		normal = glm::vec3(normal.x - 1.0f, normal.y - 1.0f, normal.z - 1.0f);
		return normal;
	}

	// ONLY Used on Device side
	__host__ __device__
	glm::vec3 getEnvironmentColor(glm::vec3& normalized_dir) {
		// Convert (normalized) dir to spherical coordinates.	

		float phi = std::atan2(normalized_dir.z, normalized_dir.x);
		phi = (phi < 0.f) ? (phi + 2.f * Pi) : phi;

		float theta = glm::acos(normalized_dir.y);
		
		glm::vec2 uv = glm::vec2(phi * Inv2Pi, 1.0f - theta * InvPi);
		
		return getColor(uv);
	}

};