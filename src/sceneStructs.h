#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "image.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

class AABB
{
public:
	AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max), center((min + max) * .5f) {}
	AABB() : min(glm::vec3(0.f)), max(glm::vec3(0.f)), center((min + max) * .5f) {}

	// Intersection is handled in device
	AABB Encapsulate(AABB bounds);
	AABB Transform(const glm::mat4x4& transform);

	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 center;

private:
	static glm::vec3 aabb[];
};

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
	glm::vec3 p1;
	glm::vec3 p2;
	glm::vec3 p3;

	// Normals
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec3 n3;

	AABB bounds;
};

struct CompactTriangle
{
	// Data
	float e1x;
	float e1y;
	float e1z;

	float e2x;
	float e2y;
	float e2z;

	float p1x;
	float p1y;
	float p1z;

	// Normals
	float n1x;
	float n1y;
	float n1z;

	float n2x;
	float n2y;
	float n2z;

	float n3x;
	float n3y;
	float n3z;
};

struct StackData
{
	int nodeOffset;
	float minDistance;
	float maxDistance;
};

struct CompactNode
{
	int leftNode;
	int rightNode;
	float split;

	// These two could be an union
	int axis;
	int primitiveCount;
};

class Mesh
{
private:
	struct MeshNode
	{
	public:
		MeshNode(const std::vector<Triangle *> &triangles, glm::vec3 min, glm::vec3 max, int depth, int maxDepth, int threshold);
		~MeshNode();

		void BuildNode(const std::vector<Triangle *> &triangles, const glm::vec3& minVector, const glm::vec3& maxVector, int depth, int maxDepth, int threshold);
		float CostFunction(float split, const std::vector<Triangle *> &triangles, float minAxis, float maxAxis);
		float GetSplitPoint(const std::vector<Triangle *> &triangles, float minAxis, float maxAxis);

		bool IsLeaf();
		int GetNodeCount();
		int TriangleCount();
		int GetDepth();

	public:
		std::vector<Triangle *> nodeTriangles;
		MeshNode * left;
		MeshNode * right;

		float split;
		int axis;
		int parentOffset; // For compaction
	};

public:
	Mesh(int maxDepth, int maxLeafSize, std::vector<Triangle*>& triangles);
	~Mesh();

	void Build();

	int maxDepth;
	AABB meshBounds;
	int maxLeafSize;
	int * compactNodes;
	int compactDataSize;

protected:
	std::vector<Triangle*> triangles;
	MeshNode * root;

	AABB CalculateAABB();
	void Compact();
};

struct MeshDescriptor
{
	int offset;
	glm::vec3 minAABB;
	glm::vec3 maxAABB;
	MeshDescriptor() : offset(-1) {};
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
