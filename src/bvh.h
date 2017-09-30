#pragma once
#include "sceneStructs.h"

// Forward declarations of structs used by our BVH tree
// They are defined in the .cpp file
struct BVHPrimitiveInfo {
	BVHPrimitiveInfo() {}
	BVHPrimitiveInfo(size_t primitiveNumber, const Bounds3f &bounds)
		: primitiveNumber(primitiveNumber),
		bounds(bounds),
		centroid(.5f * bounds.min + .5f * bounds.max) {}
	int primitiveNumber;
	Bounds3f bounds;
	glm::vec3 centroid;
};


struct BVHBuildNode {
	// BVHBuildNode Public Methods
	void InitLeaf(int first, int n, const Bounds3f &b) {
		firstPrimOffset = first;
		nPrimitives = n;
		bounds = b;
		children[0] = children[1] = nullptr;
	}
	void InitInterior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
		children[0] = c0;
		children[1] = c1;
		bounds = Union(c0->bounds, c1->bounds);
		splitAxis = axis;
		nPrimitives = 0;
	}
	Bounds3f bounds;
	BVHBuildNode *children[2];
	int splitAxis, firstPrimOffset, nPrimitives;
};

//struct MortonPrimitive {
//	int primitiveIndex;
//	unsigned int mortonCode;
//};
//
//struct LBVHTreelet {
//	int startIndex, nPrimitives;
//	BVHBuildNode *buildNodes;
//};

struct LinearBVHNode {
	Bounds3f bounds;
	union {
		int primitivesOffset;   // leaf
		int secondChildOffset;  // interior
	};
	unsigned short nPrimitives;  // 0 -> interior node, 16 bytes
	unsigned char axis;          // interior node: xyz, 8 bytes
	unsigned char pad[1];        // ensure 32 byte total size
};


struct BucketInfo {
	int count = 0;
	Bounds3f bounds;
};



LinearBVHNode* ConstructBVHAccel(//LinearBVHNode *bvh_nodes, //Output
					   int& totalNodes, //Output
					   std::vector<Triangle> &primitives, //Reshuffled according to BVH
					   int maxPrimsInNode = 1); // Adjust this to reach best performance

void DeconstructBVHAccel(LinearBVHNode *bvh_nodes);

