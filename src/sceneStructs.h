#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType 
{
    SPHERE,
    CUBE,
};

struct Ray 
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom 
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum BxDFType {
	BSDF_LAMBERT = 1 << 0,      // This BxDF represents diffuse energy scattering, which is uniformly random
	BSDF_SPECULAR_BRDF = 1 << 1,     // This BxDF handles specular energy scattering, which has no element of randomness
	BSDF_SPECULAR_BTDF = 1 << 2, 
	BSDF_ALL = BSDF_LAMBERT | BSDF_SPECULAR_BRDF | BSDF_SPECULAR_BTDF
};

struct Material 
{
    glm::vec3 color;
	float emittance;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;

	bool transmissive;
	bool reflective;
	bool refractive;

	float eta; // The ratio of indices of refraction at this surface point. Irrelevant for opaque surfaces.    

	int numBxDFs; // How many BxDFs this BSDF currently contains (init. 0)
	const static int MaxBxDFs = 8; // How many BxDFs a single BSDF can contain
	BxDFType bxdfs[MaxBxDFs]; // The collection of BxDFs contained in this BSDF

	// The PBRT BSDF stores normal and tangent data from the intersection that created it
	// Our path tracer will pre-transform vectors before they've been passed into f()
	glm::mat3 worldToTangent; // Transforms rays from world space into tangent space,
							  // where the surface normal is always treated as (0, 0, 1)
	glm::mat3 tangentToWorld; // Transforms rays from tangent space into world space.
							  // This is the inverse of worldToTangent (incidentally, inverse(worldToTangent) = transpose(worldToTangent))
	Normal3f normal;          // May be the geometric normal OR the shading normal at the point of intersection.
							  // If the Material that created this BSDF had a normal map, then this will be the latter.
};

struct Camera 
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState 
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment 
{
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection 
{
  float t;
  glm::vec3 intersectPoint;
  glm::vec3 surfaceNormal;
  int materialId;
};