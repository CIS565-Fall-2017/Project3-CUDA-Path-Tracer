#pragma once


#include <vector>
#include <glm\glm.hpp>
#include "scene.h"

struct OctreeNodeGPU {
	int parentIdx;
	int childStartIdx;
	int childEndIdx; // will prune empty nodes, therefore any number between 0, 8 on GPU
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

void treeDeleteCPUNodes(OctreeNodeCPU* node);

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

	~OctreeBuilder() {
		if (root) {
			treeDeleteCPUNodes(root);
		}	
	}
	
	void clearCPUData() {
		if (root) {
			treeDeleteCPUNodes(root);
		}
	}

	glm::vec3 sceneCenter() { return 0.5f * (minXYZ + maxXYZ); }
	glm::vec3 sceneHalfEdgeSize() { return 0.5f * (maxXYZ - minXYZ); }

	void buildFromScene(Scene* scene);
	// the data structure in 1d for GPU traversal
	std::vector<OctreeNodeGPU> allGPUNodes;
	// mesh indices in the same order as containing octree nodes
	std::vector<int> octreeOrderGeomIDX;

	void buildGPUfromCPU();
};