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

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
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
};
//Summary state has a digit for each material type
//  2^0  lambert (1) or not
//  2^1  Reflective(1) or not
//  2^2  Transmissive(1) or not
//  2^3  Emmissive (1) or not(0)
// Material can be any combination of these with the following exception:
//  The material is either a Lambert material or a Emissive material but not 
//  both.  This is because there is only one field for the Color and if it is non
//  zero and Emissive is >0 it is emissive; if it is nonzero and emissive is not 
// there it is Lambert.
// indexOfRefraction is the ratio of the index of refraction of the side away from the normal to
// the that on the side that the normal points to.:
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
    int   summaryState;
};


enum MaterialType { Lambert = 1 << 0, Reflective = 1 << 1, Refractive = 1 << 2, Emissive = 1 << 3};
// MaxMaterialTypes is number of material categories.
const  unsigned MaxMaterialTypes {4};
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
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
