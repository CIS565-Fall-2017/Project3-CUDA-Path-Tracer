#pragma once
#include "mesh.h"
//class Shader;

//std
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;


const enum class AXIS { X = 0, Y = 1, Z = 2 };
struct AABB {
	//AABB corners
	glm::vec3 min;//lower left back (model space x,y,z mins)
	glm::vec3 max;//top right front (model space x,y,z maxes)
	//AABB(glm::vec3& min, glm::vec3& max) :min(min), max(max) {}
	//AABB() :min(glm::vec3(INFINITY, INFINITY, INFINITY)), max(glm::vec3(-INFINITY, -INFINITY, -INFINITY)) {}
	//void GrowAABB(const AABB& aabb) {
	//	min = glm::min(min, aabb.min);
	//	max = glm::max(max, aabb.max);
	//}
	AABB(const glm::vec3& min, const glm::vec3& max);
	AABB();
	void GrowAABB(const AABB& aabb);
	void GrowAABBFromCentroid(const glm::vec3& centroid);
	void AddMargin();
	void MakeAABBFromTriangleIndices(const Model& model, const uint32_t mIndicesStartIndex);
	float GetComparableSurfaceArea() const ;
	AXIS GetSplitAxis() const ;
	glm::vec3 GetCentroidFromAABB() const;

};

//contiguous array of these are needed for construction, sorted into left and right children by partitioning
//can throw away the aabb and centroid info when finished. For animation, might want to hold onto them.
struct TriangleBVHData {
	int32_t id;// reference to triangle indices triplet set in mIndices from the Model reference passed in to BuildBVH
	AABB aabb;
	glm::vec3 centroid;
	uint32_t bin;//used to store the bin the triangle falls into during SAH calculation
	TriangleBVHData();
};

struct BVHNode {
	//total node should fit on a cache line (16 32bit items or 64 bytes)
	//so that when we fetch the next child node we'll have all the data we need, no need to fetch again
	//actually contains the bounds of each child
	//store the root AABB as a member of the wrapper class
	AABB leftAABB;
	AABB rightAABB;


	//in unions all members occupy the same memory space
	//Size of a union is taken according the size of largest member in union. 
	union {
		struct {
			uint32_t leftIdx; 
			uint32_t rightIdx;
		} inner;

		struct {
			uint32_t numTriangles;// if MSB is 1 then node is leaf so use leaf form of the union
			uint32_t startIdx;
		} leaf;
	} payload;
};

class BVH {
public://data
	std::vector<BVHNode> mBVHNodes;
	std::vector<uint32_t> mTriangleIDs;
	AABB localRootAABB;//The AABB of the root node in local space (must be aligned with local model axes)
	AABB worldRootAABB;//The AABB of the root in world space (must be aligned with world axes)
	uint32_t maxDepth;

	//for sbvh, will need to allocate addtional mem for when it decides to perform a spatial split to prevent excessive overlap between children of a node
	//recommended cutoff is 30% of nodes have spatial split (if neccessary)
	//so allocate up to this amount

public://functions
	BVH();

	void BuildBVH(const Model& model);

	uint32_t RecurseBuildBVH(std::vector<TriangleBVHData>& triangleBvhData, uint32_t startIdx, uint32_t onePastEndIdx, const AABB& nodeCentroidAABB, 
		const uint32_t nodeAllocIdx, uint32_t& allocIndex, uint32_t currentDepth, uint32_t& totalInnerNodes, uint32_t& totalLeafNodes);

	uint32_t CreateLeafNode(const uint32_t startIdx, const uint32_t nodeAllocIdx,
		const uint32_t currentDepth, uint32_t& totalLeafNodes, const uint32_t numTriangles);
};