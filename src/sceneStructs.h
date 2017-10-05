#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define  MAX_OCTREE_CELL 100
#define  KDTREE_MAX_STACK 128

typedef float Float;
typedef glm::vec3 Color3f;
typedef glm::vec3 Point3f;
typedef glm::vec3 Normal3f;
typedef glm::vec2 Point2f;
typedef glm::ivec2 Point2i;
typedef glm::ivec3 Point3i;
typedef glm::vec3 Vector3f;
typedef glm::vec2 Vector2f;
typedef glm::ivec2 Vector2i;
typedef glm::mat4 Matrix4x4;
typedef glm::mat3 Matrix3x3;

// Global constants. You may not end up using all of these.
#define ShadowEpsilon 0.0001f
#define RayEpsilon 0.000005f
#define RayMarchingEpsilon 0.1f
#define Pi 3.14159265358979323846f
#define TwoPi 6.28318530717958647692f
#define InvPi 0.31830988618379067154f
#define Inv2Pi 0.15915494309189533577f
#define Inv4Pi 0.07957747154594766788f
#define PiOver2 1.57079632679489661923f
#define PiOver4 0.78539816339744830961f
#define Sqrt2 1.41421356237309504880f
#define OneMinusEpsilon 0.99999994f

enum GeomType {
    SPHERE,
    CUBE,
	PLANE,
	MESH,
	TRIANGLE	
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB
{
	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 mid;
};

struct Image
{
	int ID;
	int width;
	int height;
	int beginIndex;
};



struct Octree
{
	int ID;

	bool bLeaf;
	int size;
	int firstElement;

	int child_01;
	int child_02;
	int child_03;
	int child_04;
	int child_05;
	int child_06;
	int child_07;
	int child_08;
	
	int ParentID;

	AABB boundingBox;

	//bool bTraversed;
};

struct Triangle {
	glm::vec3 position0;
	glm::vec3 position1;
	glm::vec3 position2;

	glm::vec3 normal0;
	glm::vec3 normal1;
	glm::vec3 normal2;

	glm::vec2 texcoord0;
	glm::vec2 texcoord1;
	glm::vec2 texcoord2;

	glm::vec3 planeNormal;

	int triangleID;
	int nextTriangleID;
	AABB boundingBox;
};

struct KDtreeNodeForGPU
{
	int ID;
	bool bLeaf;

	int ParentID;
	AABB boundingBox;

	int LeftID;
	int RightID;

	int size;

	int TriangleArrayIndex;

	bool bLeftTraversed;
	bool bRightTraversed;

	bool bLeftIntersected;
	bool bRightIntersected;

	float minT;
	float maxT;
};

struct KDtreeNode
{
	int ID;
	bool bLeaf;

	int ParentID;
	AABB boundingBox;

	int LeftID;
	int RightID;

	int size;

	std::vector<Triangle> triangles;
};

struct Mesh {
	int size;
	int triangleBeginIndex;
	//AABB boundingBox;
	int OctreeID;
	int KDtreeID;
};

struct Geom {
    enum GeomType type;
    int materialid;
	int ID;
	
	Mesh meshInfo;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
	glm::mat4 rotationMat;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

	AABB boundingBox;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
	float Roughness;
    float indexOfRefraction;
    float emittance;

	int diffuseTexID;
	int specularTexID;
	int normalTexID;
	int roughnessTexID;
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
	float focalDistance;
	float lensRadious;

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
	glm::vec3 accumulatedColor;
	glm::vec3 throughputColor;
	int pixelIndex;
	int remainingBounces;
	bool specularBounce;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 intersectPoint;
  glm::vec2 uv;
  glm::mat4 ratationMat;
  glm::mat4 IntersectedInvTransfrom;
  int geomId;
  int materialId;
};
