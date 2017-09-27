#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
};
enum PathType {
	LAMBERT,
	MIRROR,
	GLASS,
	TERMINATE,
};

class Ray {
public:
    glm::vec3 origin;
    glm::vec3 direction;
};

class Geom {
public:
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};
class Material {
public:
	glm::vec3 color;
	struct {
		float exponent;
		glm::vec3 color;
	} specular;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float emittance;
};

class Camera {
public:
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

class RenderState {
public:
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

class PathSegment {
public:
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
class ShadeableIntersection {
public:
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::mat3 WorldToTangent;
  glm::mat3 TangentToWorld;
};
