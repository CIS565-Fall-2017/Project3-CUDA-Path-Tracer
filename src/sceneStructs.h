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
// #define ENABLE_DIR_LIGHTING


// MIS is NOT finished yet Just ignore it!!!!!!!
// Should uncomment dir lighting first to uncomment mis
// #define ENABLE_MIS_LIGHTING


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


struct Light
{
	Geom geom;
	int geomIdx;
	float SurfaceArea;
	float selectedProb; // based on surface area
	glm::vec3 emittance;
	__host__ __device__ glm::vec3 sample(const float& x, const float& y, float* pdf) const {
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
		(*pdf) = 1.0f / 9.0f;
		return glm::vec3(geom.transform * glm::vec4(pOnObj, 1.0f));
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

  // Sort our pathSegments and related ShadeableIntersections by materialTypeID
  // Used in thrust::sort_by_key
  bool operator<(const ShadeableIntersection& that) const { return this->materialId < that.materialId; }
  bool operator>(const ShadeableIntersection& that) const { return this->materialId > that.materialId; }
};


struct RadixSortElement {
	int OriIndex;
	int materialId;
};


struct Triangle {
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
			isect->uv = uvs[0] * baryPosition.x +
						uvs[1] * baryPosition.y +
					    uvs[2] * (1.0f - baryPosition.x - baryPosition.y);

			isect->surfaceNormal= normals[0] * baryPosition.x +
								  normals[1] * baryPosition.y +
								  normals[2] * (1.0f - baryPosition.x - baryPosition.y);
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