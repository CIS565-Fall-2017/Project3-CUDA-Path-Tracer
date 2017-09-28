#include "Octree.h"

AxisBoundingBox geomBoundingBox(Geom g) {
	AxisBoundingBox abb;
	std::vector<glm::vec4> cubePts;
	cubePts.push_back(g.transform * glm::vec4(0.5, 0.5, 0.5, 1.0));
	cubePts.push_back(g.transform * glm::vec4(0.5, 0.5, -0.5, 1.0));
	cubePts.push_back(g.transform * glm::vec4(0.5, -0.5, 0.5, 1.0));
	cubePts.push_back(g.transform * glm::vec4(0.5, -0.5, -0.5, 1.0));
	cubePts.push_back(g.transform * glm::vec4(-0.5, 0.5, 0.5, 1.0));
	cubePts.push_back(g.transform * glm::vec4(-0.5, 0.5, -0.5, 1.0));
	cubePts.push_back(g.transform * glm::vec4(-0.5, -0.5, 0.5, 1.0));
	cubePts.push_back(g.transform * glm::vec4(-0.5, -0.5, -0.5, 1.0));
	glm::vec3 minimums = glm::vec3(FLT_MAX);
	glm::vec3 maximums = glm::vec3(FLT_MIN);
	for (int i = 0; i < 8; i++) {
		glm::vec4 pos = cubePts[i];
		minimums.x = minimums.x < pos.x ? minimums.x : pos.x;
		minimums.y = minimums.y < pos.y ? minimums.y : pos.y;
		minimums.z = minimums.z < pos.z ? minimums.z : pos.z;
		maximums.x = maximums.x > pos.x ? maximums.x : pos.x;
		maximums.y = maximums.y > pos.y ? maximums.y : pos.y;
		maximums.z = maximums.z > pos.z ? maximums.z : pos.z;
	}
	abb.minXYZ = minimums;
	abb.maxXYZ = maximums;
	return abb;
}

// Given a bounding box, check if it does not fit into an octant.
// Geometries that cant split the tree are given to parent nodes.
bool OctreeBuilder::overDivision(OctreeNodeCPU* node, AxisBoundingBox abb) {
	glm::vec3 min = abb.minXYZ;
	glm::vec3 max = abb.maxXYZ;
	glm::vec3 center = node->center;

	return (min.x < center.x && max.x > center.x) ||
		(min.y < center.y && max.y > center.y) ||
		(min.z < center.z && max.z > center.z);
}

// Get the octant of the geometry's center point.
// Do this after checking whether a geometry fits in an octant.
// Returns a 3 bit number corresponding to +- xyz octant
int OctreeBuilder::octant(OctreeNodeCPU* node, Geom g) {
	glm::vec3 center = g.translation;
	glm::vec3 nodeCenter = node->center;
	int oct = 0;
	if (center.z > nodeCenter.z) oct += 1;
	if (center.y > nodeCenter.y) oct += 1 << 1;
	if (center.x > nodeCenter.x) oct += 1 << 2;
	return oct;
}

// split a leaf into octants, assign existing geometries accordingly
void OctreeBuilder::splitNode(OctreeNodeCPU* node, Scene* scene) {
	if (node->depth + 1 > maxDepth) {
		// must limit size for traversal on GPU.
		return;
	}
	if (node->children.size()) {
		return; // should never happen
	}
	if (node->eltIndices.size()) {
		return; // should never happen
	}

	// check if any contained geometry can actually go inside an octant
	std::vector<bool> fits = std::vector<bool>();
	for (int i = 0; i < node->eltIndices.size(); i++) {
		fits.push_back(overDivision(node, scene->geomBounds[node->eltIndices[i]]));
	}
	bool allFit = true;
	for (int i = 0; i < fits.size(); i++) {
		allFit = allFit && fits[i];
	}

	if (allFit) {
		// no pieces of geometry can fit into octants. do not divide
		return;
	}

	glm::vec3 childHalfSideLength = 0.5f * node->halfSideLength;

	for (int i = 0; i < 8; i++) {
		OctreeNodeCPU* child = new OctreeNodeCPU;
		child->depth = node->depth + 1;
		child->halfSideLength = childHalfSideLength;
		child->center = node->center
			+ glm::vec3(
				((i >> 2) & 1) ? 1 : -1, 
				((i >> 1) & 1) ? 1 : -1, 
				(i & 1) ? 1 : -1) 
			* childHalfSideLength;
		child->parent = node;
		node->children.push_back(child);
	}

	// increment the total number of nodes. will need for GPU translation later
	numNodes += 8;

	std::vector<int> keep = std::vector<int>();

	for (int i = 0; i < node->eltIndices.size(); i++) {
		int idx = node->eltIndices[i];
		if (fits[i]) {
			keep.push_back(idx);
		}
		else {
			OctreeNodeCPU* child = node->children[octant(node, scene->geoms[idx])];
			child->eltIndices.push_back(idx);
		}
	}

	node->eltIndices = keep;
}

void OctreeBuilder::buildFromScene(Scene* scene) {
	if (scene == NULL) return;

	// make the bounding boxes for each geometry in this scene
	scene->geomBounds = std::vector<AxisBoundingBox>();
	for (int i = 0; i < scene->geoms.size(); i++) {
		scene->geomBounds.push_back(geomBoundingBox(scene->geoms[i]));
	}

	minXYZ = glm::vec3(FLT_MAX);
	maxXYZ = glm::vec3(FLT_MIN);

	// get the bounds of the entire scene
	for (int i = 0; i < scene->geomBounds.size(); i++) {
		glm::vec3 minimum = scene->geomBounds[i].minXYZ;
		glm::vec3 maximum = scene->geomBounds[i].maxXYZ;
		minXYZ.x = minXYZ.x > minimum.x ? minimum.x : minXYZ.x;
		minXYZ.y = minXYZ.y > minimum.y ? minimum.y : minXYZ.y;
		minXYZ.z = minXYZ.z > minimum.z ? minimum.z : minXYZ.z;
		maxXYZ.x = maxXYZ.x < maximum.x ? maximum.x : maxXYZ.x;
		maxXYZ.y = maxXYZ.y < maximum.y ? maximum.y : maxXYZ.y;
		maxXYZ.z = maxXYZ.z < maximum.z ? maximum.z : maxXYZ.z;
	}

	glm::vec3 sceneCenter = 0.5f * (maxXYZ + minXYZ);
	glm::vec3 halfSideLength = 0.5f * (maxXYZ - minXYZ);

	numNodes = 1;
	root = new OctreeNodeCPU;
	root->center = sceneCenter;
	root->halfSideLength = halfSideLength;
	root->depth = 0;
	root->parent = NULL;

	// iterate through all bounding boxes. determine whether to assign to an existing node or split it
	for (int i = 0; i < scene->geomBounds.size(); i++) {
		OctreeNodeCPU* current = root;
		bool cannotFit = false;
		// traverse existing nodes until cannot fit into octant, or found a leaf
		while (current->children.size() > 0) {
			if (overDivision(current, scene->geomBounds[i])) {
				break;
			}
			auto* child = current->children[octant(current, scene->geoms[i])];
			current = child;
		}
		// either reached a leaf or the geometry cannot fit into an octant of current
		// if reached a leaf, cannot yet tell if does not fit into octant.
		// add it to current regardless.
		current->eltIndices.push_back(i);

		if (current->children.size() == 0 && current->eltIndices.size() > 1) {
			// special case: just added to a non-empty leaf. 
			// try to divide it. if none of the contained geoms can fit in an octree, 
			// it will not divide. this is handled in splitNode.
			splitNode(current, scene);
		}
	}
}