#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType 
{
    SPHERE,
    CUBE,
	TRIANGLE
};

enum BxDFType {
	BSDF_LAMBERT = 1 << 0,      // This BxDF represents diffuse energy scattering, which is uniformly random
	BSDF_SPECULAR_BRDF = 1 << 1,     // This BxDF handles specular energy scattering, which has no element of randomness
	BSDF_SPECULAR_BTDF = 1 << 2,
	GLASS = 1 << 3,
	BSDF_ALL = BSDF_LAMBERT | BSDF_SPECULAR_BRDF | BSDF_SPECULAR_BTDF | GLASS
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

struct Material 
{
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

	int numBxDFs; // How many BxDFs this BSDF currently contains (init. 0)
	const static int MaxBxDFs = 8; // How many BxDFs a single BSDF can contain
	BxDFType bxdfs[MaxBxDFs]; // The collection of BxDFs contained in this BSDF(in our implementation there isn't a
							  //practical difference between a material and a BSDF)
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

	float lensRadius; //Depth of Field Parameter
	float focalDistance; //Depth of Field Parameter
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
