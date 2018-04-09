#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "model.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	PLANE,
	MODEL,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct ModelInfo {
	Model* model = nullptr;
	AABB worldRootAABB = AABB();//check this first, if hit begin fetching nodes
	uint32_t bvhMaxDepth = 0;//informs the max stack size for pushing and popping of bvhNode indexes during traversal on the gpu
	//Corresponding Start Index for the 3 contiguous arrays for Model data
	//NOTE: might be worth normalizing the index pointers as the model data is looped through and compacted into contiguous arrays
	//then sent to the gpu.This will avoid additional sums with the startIdx's to calculate correct array position 
	uint32_t startIdxBVHNode = 0;//bvh node array
	uint32_t startIdxTriIndex = 0;//triangle indices array
	uint32_t startIdxTriVertex = 0;//triangle vertex array
};

struct Geom {
    enum GeomType type;
    int materialid;
	ModelInfo modelInfo;
    glm::vec3 translation;//really store this?
    glm::vec3 rotation;//really store this?
    glm::vec3 scale;//really store this?
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat3 invTranspose;//transpose the inverse on the fly or store this? //only transforming normals with this so no need for translation (mat4)
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float hasSubSurface;
    float indexOfRefraction;
    float emittance;
    glm::vec3 sigA;
    glm::vec3 sigSPrime;

//jensen page 5 top right
//marble: sigA(units are per mm) = 0.0021, 0.0041, 0.0071 
//marble: sigSPrime = 2.19, 2.62, 3
//marble: eta = 1.5
//marble: diffuse reflectance should be = 0.83, 0.79, 0.75 

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
	int remainingBounces;
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	glm::ivec2 MSPaintPixel;//debugging
	glm::vec3 throughput;
	bool specularbounce;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int geomId;
};

