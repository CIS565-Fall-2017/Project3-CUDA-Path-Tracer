#pragma once
#include "mesh.h"
//class Shader;

//std
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

struct AABB {
	//AABB corners
	glm::vec3 min;//lower left back (model space x,y,z mins)
	glm::vec3 max;//top right front (model space x,y,z maxes)
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
			uint32_t startIndex;
		} leaf;
	} payload;
};

class BVH {
public://data
	std::vector<BVHNode> mBVHNodes;
	const Model& model;
	AABB localRootAABB;
	AABB worldRootAABB;

	//for sbvh, will need to allocate addtional mem for when it decides to perform a spatial split to prevent excessive overlap between children of a node
	//recommended cutoff is 30% of nodes have spatial split (if neccessary)
	//so allocate up to this amount

public://functions
	BVH(const Model& model);
	void BuildBVH();
};

