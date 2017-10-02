#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "image.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType 
{
    SPHERE,
    CUBE,
	MESH,
};

struct Ray 
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle
{
	// Data
	glm::vec3 e1;
	glm::vec3 e2;
	glm::vec3 p1;

	// Normals
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec3 n3;
};

struct Mesh
{
	std::vector<Triangle> triangles;
};

struct MeshDescriptor
{
	int offset;
	int triangleCount;
	MeshDescriptor() : offset(-1), triangleCount(0) {};
};

struct Geom 
{
    enum GeomType type;
    int materialid;
	MeshDescriptor meshData;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct TextureDescriptor
{
	int valid;
	int type; // 0 bitmap, 1 procedural
	int index;
	int width;
	int height;
	glm::vec2 repeat;
	TextureDescriptor() : index(-1), width(0), height(0), repeat(glm::vec2(1.f)), type(0), valid(-1) {};
};

struct Material
{
	glm::vec3 color;
	struct
	{
		float exponent;
		glm::vec3 color;
	} specular;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float emittance;
	float translucence;
	TextureDescriptor diffuseTexture;
	TextureDescriptor specularTexture;
	TextureDescriptor normalTexture;
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
	float aperture;
	float focalDistance;
	TextureDescriptor bokehTexture;
};

struct Film 
{
	float filterRadius;
	float filterAlpha;
	float invGamma;
	float gamma;
	float exposure;
	float vignetteStart;
	float vignetteEnd;

	Film() : filterRadius(1.5f), filterAlpha(4.0f), gamma(2.2f), invGamma(1.f / 2.2f), exposure(16.f), vignetteStart(.3f), vignetteEnd(1.25f) {};
};

struct RenderState 
{
    Camera camera;
	Film film;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec4> image;
    std::string imageName;
};

struct PathSegment 
{
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// The prefiltered sample that contains the color and offset
struct SampledPath 
{
	glm::vec3 color;
	glm::vec2 position;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
	float t;
	int materialId;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
};
