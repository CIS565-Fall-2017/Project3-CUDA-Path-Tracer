#pragma once


#include <vector>
#include <glm\glm.hpp>
#include "scene.h"

struct OctreeNodeGPU {
	int parentIdx;
	int childStart;
	int eltStartIdx;
	int eltEndIdx;
	int depth;
	glm::vec3 center; // size determined at runtime
};


typedef struct OctreeNodeCPU {
	OctreeNodeCPU* parent;
	glm::vec3 center;
	glm::vec3 halfSideLength;
	int depth;
	std::vector<int> eltIndices = std::vector<int>();
	std::vector<OctreeNodeCPU*> children = std::vector<OctreeNodeCPU*>();
} OctreeNodeCPU;

class OctreeBuilder {
private:
	int numNodes;
	int maxDepth;
	OctreeNodeCPU* root = NULL;
	glm::vec3 minXYZ;
	glm::vec3 maxXYZ;
	bool valid;
public:
	// check if the geometry does not fit into a partition 
	bool overDivision(OctreeNodeCPU* node, AxisBoundingBox abb);
	void addEltToNode(OctreeNodeCPU* node, int geomIdx) {
		node->eltIndices.push_back(geomIdx);
	}

	void splitNode(OctreeNodeCPU* node, Scene* scene);

	// returns number octant of geometry centerpoint. -1 if not in node,
	// otherwise bit corresponds to 0 (-) 1 (+) x, y, z e.g. 3 is -x, +y, +z
	int octant(OctreeNodeCPU* node, Geom g);

	// Initialize with a scene
	OctreeBuilder() {
		maxDepth = 9;
		numNodes = 0;
		valid = false;
	}

	void buildFromScene(Scene* scene);

	// once the tree is built, make parallel arrays for GPU usage
	OctreeNodeGPU* convertCPUToGPU(int** octreeOrderedGeomIdx, int* numNodes);
};