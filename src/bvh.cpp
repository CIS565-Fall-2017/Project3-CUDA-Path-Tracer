#pragma once
#include "bvh.h"
#include "model.h"
#include "mesh.h"


//glm
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


//SOIL
#include <SOIL2/SOIL2.h>

//////external
//#include <stb/stb_image.h>

#include <algorithm>
#include <sstream>
#include <iostream>
#include <array>


TriangleBVHData::TriangleBVHData()
	: id(-1), aabb(), centroid(glm::vec3(INFINITY, INFINITY, INFINITY)), bin(-1)
{

}
	


AABB::AABB(const glm::vec3& min, const glm::vec3& max) 
	: min(min), max(max) 
{
}

AABB::AABB() 
	: min(glm::vec3(INFINITY, INFINITY, INFINITY)), max(glm::vec3(-INFINITY, -INFINITY, -INFINITY))
{
}

void AABB::GrowAABB(const AABB& aabb) {
	min = glm::min(min, aabb.min);
	max = glm::max(max, aabb.max);
}

void AABB::AddMargin() {
	constexpr float margin = 0.005;
	min -= glm::vec3( margin,  margin,  margin);
	max += glm::vec3( margin,  margin,  margin);
}

void AABB::GrowAABBFromCentroid(const glm::vec3& centroid) {
	min = glm::min(min, centroid);
	max = glm::max(max, centroid);
}

glm::ivec3 AABB::MakeAABBFromTriangleAndGetIndices(const Model& model, const uint32_t idx1) {
	glm::ivec3 vertIndices;
	vertIndices[0] = model.mIndices[idx1 + 0];
	vertIndices[1] = model.mIndices[idx1 + 1];
	vertIndices[2] = model.mIndices[idx1 + 2];

	const glm::vec3& p1 = model.mVertices[vertIndices[0]].pos;
	const glm::vec3& p2 = model.mVertices[vertIndices[1]].pos;
	const glm::vec3& p3 = model.mVertices[vertIndices[2]].pos;

	min = glm::min(glm::min(p1, p2), p3);
	max = glm::max(glm::max(p1, p2), p3);
	
	return vertIndices;
}

glm::vec3 AABB::GetCentroidFromAABB() const {
	return 0.5f * (min + max);
}

float AABB::GetComparableSurfaceArea() const {
	const float lenX = max.x - min.x;
	const float lenY = max.y - min.y;
	const float lenZ = max.z - min.z;
	return lenX*lenY + lenX*lenZ + lenY*lenZ;
}

AXIS AABB::GetSplitAxis() const {
	const float lenX = max.x - min.x;
	const float lenY = max.y - min.y;
	const float lenZ = max.z - min.z;
	const float max = std::max(std::max(lenX, lenY), lenZ);
	if (lenX == max) {
		return AXIS::X;
	} else if (lenY == max) {
		return AXIS::Y;
	} else if (lenZ == max) {
		return AXIS::Z;
	}
}

BVH::BVH() 
{
}

void BVH::BuildBVH(const Model& model) {
	/* NOTES: On fast construction of SAH-based Bounding Volume Hierarchies. Ingo Wald
		* SA (surface area) is actually the total surface area of the bounding box, not just the the face we care about
		* this paper has a node occupying half a cache line (32 bytes) storing the bounding box of the node
			some flags and the union section is either both child nodes indices or triangleID array start index and number of triangles,
			depending on whether it's an inner or leaf node.
		* when preparing to split a node, form AABB around centroids of triangles
			- There are K equidistant spaced partitions along the larges axis (checking
			 other axes doesn't seem to be worth the effort). K-1 positions along the axis to split.
			- Bins count the triangles that overlap it( or centroids that fall into it)
			- Bins keep track of the bin bounds: the bounding box of the triangles that touch it. Could be larger or smaller than the 'domain' of the actual bin.
			- Full 3D bounds of these bind bounds are kept track of. not just the 2d face along the axis we are splitting
			- Cost Function: NumLeft*AreaLeft ++ NumRight*AreaRight

		* In the setup, compute each triangles bounds and its centroid. Also compute total bounds and total centroid bounds.
			-  for SIMD friendly format store the 3D positions of the above info in four 4-byte aligned floats (16 bytes total, 128 bits total)
			- setup stage requires 14 SSE ops per triangle. 3 loads for tri verts, two mins and maxes each for tri bounds
			 one min and max each to grow voxel bounds, one add and one mul0.5f for the centroid another min and max each for growing the centroid
			 and 3 stores to write tri bounds and centroid to memory
		
		* Triangle-to-bin projection: simply get the percentage along the centroid bounds splitting axis the triangle's centroid is,
		then scale it by the number of bins times 1.f-epsilon to get the floating point bin number. Then perform float to int truncation to get the bin number. 
			- initialize each bin to a negative box [+inf, -inf], allowing it to grow to include a triangles bound with one SSE min and one max
			  without having to check if empty.
			- can later merge the individual bin bounds without having to check for emptiness since growing an AABB with neg AABB using min max ops does not 
			  change the original box. 
			- Don't have to use all the tri's, just a subset but this requires param tuning to not degrade the bvh quality.
		
		* SAH evaluation: first do a linear pass from the left where you incrementally accum the bounds and num of tris for the left half. 
			storing these two values (num tris and total tris 3d bound surface area for each splitting plane (also no need to mult each face by 2 for total area since 
			this divides out when comapring a Surface area vs another surface area, similar to not taking the sqrt when comparing distances, just compare the squared distances))
			Then do the same for the right and evaluate the SAH for each plane. keep track of the min cost splitting plane
			if one of the sides of a splitting plane does not have any tris then it gets rejected as a candidate as BVH is not allowed empty partitions
		
		* In-place ID list partitioning: bvh is two contuguous arrays one for the node data one for the triangleID 
		(ref the triangle in some way i.e. 3 vertex indices that refers to the main vertex array or simply and index to the actual index array)
			- An in place bvh for N triangles has at most 2N-1 nodes, this can be pre-allocated
			- bvh triangles are partitioned into nodes by partioning the triangleID array (uint32_t's init to 0,1,2...N)
			- when doing sbvh, will need to allocate up to spatial plit budget and create addition refs when a spatial split is neccessary.
			- can use std::partition to do the partitioning or have two iterators, one starts at the front of the triangleID array and one at the back
			 the left iter scans to the right until it finds a tri that should be on the right of the split plane, the right iter scans to the left until
			 it finds a tri that should be on the left, then swap these two triangleID's. keep going while(iterLeft < iterRight)
			 - pass on the triangle bounds and centroid bounds of the two children to the recursive calls, as well as teh range int he traingleID array that the call should operate over
			 - triangleID array should probably contain the bbox and centroid of the triangle in addition to it's id since these will be needed on every recursive call
			 - the centroid-to-bin sorting this can probably be done in-place on the triangleID struct array to prevent the need for the partition step later
		
		* Number of bins: 4-16 seems to work fine 16 being the best quality.
		This number can probably decrease as you go down the tree?
		
		* Termination: max tris per leaf: 2-4 is fine. if centroid bounds is too small...

		* Parallelization: page 5 talks about it

		* Grid based binning: bot page 5
	*/

	/* NOTES: Understanding the Efficiency of Ray Traversal on GPUs
	   NOTES: and 2012 addendum
	   * addendum suggests that modern hardware for work scheduling reduces the need for persistant threads:
	    the idea that you allocate enough threads to fill the machine. These threads stick around and fetch work to be done.

		* if hit both children procede to the child that is closer and push the index of the other on the stack
		* Uses Woop intersection test for triangles
		* uses a 1D texture to store the node array, triangle data in global mem
		* if-if seemed better than while-while
	*/



	/* NOTES: Spatial Splits in Bounding Volume Hierarchies. Martin Stich et al
		* possible in place build using techniques discussed in Wachter and Keller 2007
		* alpha of 1e-5 is best
		* use BFS building (push and pop from queue) to limit spatial splits to upper nodes if theres a memory budget(+30% is good)
		* Section 4: The SBVH:
		* Chopped binning:
			- The aabb of the triangle is used to determine which bins the triangle gets stored in. The bounds of its aabb is clipped against the bounds of the bin.
			- These clipped aabb's of the triangle are used to grow the aabb of all triangle clips within the bin.
			- Uses two counters: one to count a reference entry and one to count a reference exit.
			- To count the number of triangle refs left of a split plane simply add up the entry counters in each bin to the left of plane.
			- To count the number of triangle refs to the right of split plane, add up the exit counters in each bin to the right of the plane.
			- The cheapest split plane is compared against the best object split
		* Unsplitting a reference
			-  triangles that straddle a splitting plane are looped through and checked to see what the best option is:
				1. split the triangles bounding box across the plane and have two references, one for each child
				2. put it entirely in the left child
				3. put it entirely in the right child
			- The cost for each of these is calculate and the cheapest is chosen.
			- Cost for splitting the triangle across the plane:
				1. Csplit = AreaBoundsLeft*NumTrianglesLeft + AreaBoundsRight*NumTrianglesRight
				This is the same for all split triangles
			- Cost for putting a split triangle entirely into the left bin:
				2. Cleft = AreaBoundsLeftMergedWithEntireTriangleBounds*NumTrianglesLeft + AreaBoundsRight*(NumTrianglesRight-1)
			- Cost for putting a split triangle entirely into the right bin:
				3. Cright = AreaBoundsLeft*(NumTrianglesLeft-1) + AreaBoundsRightMergedWithEntireTriangleBounds*NumTrianglesRight
			-Finally, We want to reduce node overlap but not have memory explode due to triangle refernece duplication when they are put into both children.
			- To avoid this, the paper suggests using and alpha value:
			- First find lambda: the amount of surface area overlap of the full unsplit AABB's of each child
			 after the plane for the best object split is found (done before considering to perform spatial splits of triangles that straddle this plane)
			- To see an example see Figure 4, page 4
			- procede with performing spatial splits if alpha < lambda / surfaceAreaOfBoundOfRootNodeForTheAxisInQuestion
			- best value for alpha is 1e-5 (figure 5, page 5)
			- This alpha value represents the highest tolerated ratio of overlap area to root area without attempting spatial splits.
			- smaller nodes are unlikely to hit this threshold so splits tend to happen near the root of the tree, where it matters most and has most effectiveness
		* number of bins was 256 at all levels
		* How the sbvh improves performance vs normal bvh
			- scenes can consist of portions (such as a character) that have high poly count and finely detail and some sections like a 
			surrounding environment that isnt as detailed with a low poly count and larger triangles.
			- if you were to compare sbvh vs bvh on the high poly character there wouldnt be much of a perf boost since there wouldnt be much cases for spatial splits
			- however for non-homogenous scenes (such as a high poly character surrounded by a low poly environment) the low tesselation of the environment causes
			overlap in the important top levls of a bvh which in combination with the dense character geometry makes hierarchy traversal expensive. 
			This situation is mitigated with minimal reference duplication by the sbvh's spatial splits (when neccessary) which easily avoid top level overlap by
			placing a small number of spatial splits. see figure 1 of teh papaer or the title image
	*/



	//we need to make a temporary working set of data for each triangle's ID AABB and centroid
	//we'll need to shuffle around the mIndices array of Model adjecent triplets of indices is a triangle that must be moved together
	//when using sbvh might be worth having a separate indices array from the opengl one since there will be duplicates
	AABB localRootCentroidAABB;
	const uint32_t numTriangles = model.mIndices.size() / 3;
	std::vector<TriangleBVHData> triangleBvhData(numTriangles);

	//an in-place bvh requires 2N-1 nodes if there's no ref duplication
	mBVHNodes.reserve(2 * numTriangles - 1);//acutal mem alloc block
	mBVHNodes.resize(2 * numTriangles - 1);//sets valid range, can access with [] operator, won't run into idx out of bounds issues


	//loop through triangles and calc the triangle aabb centroids for the triangleBvhData vector
	for (int i = 0; i < numTriangles; ++i) {
		TriangleBVHData& data = triangleBvhData[i];
		data.id = i;
		data.vertIndices = data.aabb.MakeAABBFromTriangleAndGetIndices(model, i*3);
		data.centroid = data.aabb.GetCentroidFromAABB();
		localRootAABB.GrowAABB(data.aabb);
		localRootCentroidAABB.GrowAABBFromCentroid(data.centroid);
	}

	//Begin recursive building of BVH tree
	//Params: start and one-past end for the index range to operate over (triangleBvhData vector)
	//Centroid bounds of the node the call needs to split
	//bvh node array index in which to put the node data
	//stats: current depth, total nodes, total inner, total leaf
	uint32_t allocIdx = 0, startIdx = 0, currentDepth = 0, nodeAllocIdx = 0,
		totalInnerNodes = 0, totalLeafNodes = 0, onePastEndIdx = numTriangles;
	maxDepth = currentDepth;

	maxDepth = RecurseBuildBVH(triangleBvhData, startIdx, onePastEndIdx, localRootCentroidAABB, nodeAllocIdx, allocIdx, currentDepth, totalInnerNodes, totalLeafNodes);
	std::cout << "\n\nBVH stats: Max Depth: " << maxDepth;
	std::cout << "\n\t TotalTri's: " << numTriangles;
	std::cout << "\n\t DuplicateTriRefs: " << 0;
	std::cout << "\n\t TotalNodes: " << totalInnerNodes + totalLeafNodes;
	std::cout << "\n\t TotalInnerNodes: " << totalInnerNodes;
	std::cout << "\n\t TotalLeafNodes: " << totalLeafNodes;
	std::cout << "\n\t localRootAABB: { " << localRootAABB.min.x << ", " << localRootAABB.min.y << ", " << localRootAABB.min.z << " }";
	std::cout << "  { " << localRootAABB.max.x << ", " << localRootAABB.max.y << ", " << localRootAABB.max.z << " }\n";

	//pack indices into triangle indices array
	mTriangleIndices.resize(triangleBvhData.size());
	for (int i = 0; i < mTriangleIndices.size(); ++i) {
		mTriangleIndices[i] = triangleBvhData[i].vertIndices;
	}

	//copy the models triangle vertex array here 
	//long term: have either one hold all the data or make bvh a class that operates on things, members of the model class
	mVertices = model.mVertices;

	//need the worldAABB

}

uint32_t BVH::RecurseBuildBVH(std::vector<TriangleBVHData>& triangleBvhData, const uint32_t startIdx, 
	const uint32_t onePastEndIdx, const AABB& nodeCentroidAABB, const uint32_t nodeAllocIdx,
	uint32_t& allocIdx, const uint32_t currentDepth, uint32_t& totalInnerNodes, uint32_t& totalLeafNodes)
{
	//NOTE: Should all uint32_t's be uint64_t's to handle several million triangle meshes??? Does CUDA handle uint64_t addresses?
	constexpr uint32_t maxTrianglesPerNode = 4;
	constexpr uint32_t numBinsPerNode = 16;//paper said 16 was as high as you need to go, however if this is too large and the mesh is also very large could get __chkstk() failure since we allocate a std::array (goes on stack) for each recursion call
	constexpr uint32_t lastBinIdx = numBinsPerNode - 1;

	//peform failure checks
	if (startIdx >= onePastEndIdx) {
		std::cout << "\n\nBVH Error: startIdx is greater than or equal to onePastEndIdx\n";
	}

	//do early termination checks
	const uint32_t numTriangles = onePastEndIdx - startIdx;
	if (numTriangles <= maxTrianglesPerNode) { //make leaf node
		return CreateLeafNode(startIdx, nodeAllocIdx, currentDepth, totalLeafNodes, numTriangles);
	}

	//Begin sorting triangles into bins (if can't peform in-place partion here, have auxilary scratch memory vector for this part)
	//and gather Surface Area Heuristic(SAH) info in two sweep, from left to right and right to left
	struct BinInfoSAH {
		//just in this bin
		uint32_t num;
		AABB aabb;
		AABB centroidAABB;

		//left info
		uint32_t numLeft;
		AABB aabbLeft;
		AABB centroidAABBLeft;
		float SALeft;

		//right info
		uint32_t numRight;
		AABB aabbRight;
		AABB centroidAABBRight;
		float SARight;

		//final SAH
		float SAH;
		BinInfoSAH() : num(0), aabb(), centroidAABB(), 
			numLeft(0), aabbLeft(), centroidAABBLeft(), SALeft(INFINITY), 
			numRight(0), aabbRight(), centroidAABBRight(), SARight(INFINITY), SAH(INFINITY)
		{
		}
	};
	//the last one will not have its numleft and numright info updated 
	//partition info(left/right) kept in the 0 through n-1 bins, partition is the right wall of the bin
	std::array<BinInfoSAH, numBinsPerNode> binInfoSAH;

	//finds the longest axis of this node's triangle centroids AABB and selects that as the axis on which to split along 
	//by creating regularly spaced partition planes 
	const uint32_t axis = (uint32_t)nodeCentroidAABB.GetSplitAxis();
	const float axisLen = nodeCentroidAABB.max[axis] - nodeCentroidAABB.min[axis];
	constexpr float scalePercentByNumBins = (numBinsPerNode)*(1.f - FLT_EPSILON);
	const float preCalcFactor = scalePercentByNumBins / axisLen;

	//go through all triangles and determine the bin that contains their centroid, update num and aabb for that bin
	{
		for (int i = startIdx; i < onePastEndIdx; ++i) {
			const float centroidLen = (triangleBvhData[i].centroid[axis] - nodeCentroidAABB.min[axis]);
			const uint32_t binNumber = preCalcFactor * centroidLen;
			//^The way this math works out it's best to recalc the bin num during partitioning and see if its more or less than the partitionWithMinSAH
			//there are really quirky floating point issues with things lying verying close to the partition
			//doing something like:
			//const float partitionSpacing = axisLen / numBinsPerNode;
			//const float partitionAxisPos = nodeCentroidAABB.min[axis] + ((partitionWithMinSAH + 1)*partitionSpacing);
			//will yield slightly off position compared to what this formula thinks is the exact partiition position
			//in fact it might be worth storing the formulas version of the partition plane position in the binInfoSAH array
			//just ran into more floating point issues, the other option is to add another field in TriangleBvhData data for bin number

			triangleBvhData[i].bin = binNumber;
			++binInfoSAH[binNumber].num;
			binInfoSAH[binNumber].aabb.GrowAABB(triangleBvhData[i].aabb);
			binInfoSAH[binNumber].centroidAABB.GrowAABBFromCentroid(triangleBvhData[i].centroid);
		}
	}

	//sweep from left to right determining the numLeft and aabbLeft for that bin 
	//(partition plane is the right wall of the bin)
	{
		//first bin doesn't have a previous bin to merge with, do outside the loop
		binInfoSAH[0].numLeft = binInfoSAH[0].num;
		binInfoSAH[0].aabbLeft = binInfoSAH[0].aabb;
		binInfoSAH[0].centroidAABBLeft = binInfoSAH[0].centroidAABB;
		binInfoSAH[0].SALeft = binInfoSAH[0].aabbLeft.GetComparableSurfaceArea();

		for (int i = 1; i < lastBinIdx; ++i) {//dont do the last bin since it is past the last partition plane
			BinInfoSAH& curr = binInfoSAH[i];
			BinInfoSAH& prev = binInfoSAH[i - 1];
			curr.numLeft = curr.num + prev.numLeft;
			curr.aabbLeft = curr.aabb; 
			curr.aabbLeft.GrowAABB(prev.aabbLeft);
			curr.centroidAABBLeft = curr.centroidAABB;
			curr.centroidAABBLeft.GrowAABB(prev.centroidAABBLeft);
			curr.SALeft	= curr.aabbLeft.GetComparableSurfaceArea();
		}
	}

	//same as above but from right to left, also keep track of minimum SAH partition
	int32_t partitionWithMinSAH = -1;
	{
		//last-1 (last partition info, numBinsPerNode-2) bin doesn't have a previous right bin to merge with, do outside the loop
		float minSAH = FLT_MAX;
		constexpr uint32_t lastPartitionIdx = lastBinIdx - 1;
		BinInfoSAH& info = binInfoSAH[lastPartitionIdx];
		info.numRight = binInfoSAH[lastPartitionIdx + 1].num;
		info.aabbRight = binInfoSAH[lastPartitionIdx + 1].aabb;
		info.centroidAABBRight = binInfoSAH[lastPartitionIdx + 1].centroidAABB;
		info.SARight = info.aabbRight.GetComparableSurfaceArea();
		info.SAH = info.SARight * info.numRight + info.SALeft * info.numLeft;

		//keep track of minimum SAH info
		if (info.numLeft > 0 && info.numRight > 0 && info.SAH < minSAH) {
			minSAH = info.SAH;
			partitionWithMinSAH = lastPartitionIdx;
		}

		for (int i = lastPartitionIdx-1; i >= 0; --i) {
			BinInfoSAH& curr = binInfoSAH[i];
			BinInfoSAH& prev = binInfoSAH[i + 1];
			curr.numRight = prev.num + prev.numRight;//numTriangles - binInfoSAH[i].numLeft
			curr.aabbRight = prev.aabb;
			curr.aabbRight.GrowAABB(prev.aabbRight);
			curr.centroidAABBRight = prev.centroidAABB; 
			curr.centroidAABBRight.GrowAABB(prev.centroidAABBRight);
			curr.SARight	= curr.aabbRight.GetComparableSurfaceArea();
			curr.SAH = curr.SARight * curr.numRight + curr.SALeft * curr.numLeft;
			if (curr.numLeft > 0 && curr.numRight > 0 && curr.SAH < minSAH) {
				minSAH = curr.SAH;
				partitionWithMinSAH = i;
			}
		}
	}

	//have the splitting partion, now partition the triangleBvhData vector such that triangles with centroids that appear less or equal to the axis split location 
	//come before the triangles that appear greater than the axis split location, within the vector
	//NOTE: this step can probably be performed when sifting centroids into bins above

	//TODO: CLEAN THIS UP (logic), there's weird cases when axisLen is 0.f and this leads to things ending up in the same bin and
	//-1 as the partionWithMinSAH, need to detect this early then put one half the triangles in one partition and the other in the remaining
	uint32_t onePastLeftPartition = startIdx;

	//NOTE: not really needed when doing more than 1 per bin
	if (0.f == axisLen && -1 == partitionWithMinSAH) {
		//kind of commone for a mesh where the aabb and/or centroids were the same for 2 triangles, was testing 1=maxTrianglesPerNode
		//rare case, but can probably happen with more triangles in a higher maxTrianglesPerNode situation
		//axis len can be 0. for partitionWIthMinSAH can be -1 if all fall into same bin due to having the same centroid
		onePastLeftPartition = (onePastEndIdx + startIdx) >> 1;//middle index in the range we're working on
		partitionWithMinSAH = 0;//we'll override the data in this bin to facilitate this special case
		binInfoSAH[0].aabbLeft = AABB();
		binInfoSAH[0].aabbRight = AABB();
		binInfoSAH[0].centroidAABBLeft = AABB();
		binInfoSAH[0].centroidAABBRight = AABB();
		//find the total aabb and centroidAABB for triangles from startIdx to before onePastLeftPartition and save to left data in bin 0
		//do the same for triangles from onePastLeftPartition to before onePastEndIdx and save to right data in bin 0
		for (int i = startIdx; i < onePastLeftPartition; ++i) {
			binInfoSAH[0].aabbLeft.GrowAABB(triangleBvhData[i].aabb);
			binInfoSAH[0].centroidAABBLeft.GrowAABBFromCentroid(triangleBvhData[i].centroid);
		}
		for (int i = onePastLeftPartition; i < onePastEndIdx; ++i) {
			binInfoSAH[0].aabbRight.GrowAABB(triangleBvhData[i].aabb);
			binInfoSAH[0].centroidAABBRight.GrowAABBFromCentroid(triangleBvhData[i].centroid);
		}
	} else {
		uint32_t leftIter = startIdx;
		uint32_t rightIter = onePastEndIdx - 1;
		while (true) {
			//sweep left to right until find one that should be to the right of the plane
			while ( triangleBvhData[leftIter].bin <= partitionWithMinSAH && leftIter < onePastEndIdx) { ++leftIter; }
			//sweep right to left until find one that should be to the left of the plane
			while ( triangleBvhData[rightIter].bin > partitionWithMinSAH && rightIter > startIdx) { --rightIter; }

			if (leftIter > rightIter) {
				onePastLeftPartition = onePastEndIdx == leftIter ? onePastEndIdx - 1 : leftIter;//issues when there are two remaining with same centroid (should you decide to do 1 tri/bin
				break;
			} else { //swap data
				TriangleBVHData tmp = triangleBvhData[leftIter];
				triangleBvhData[leftIter] = triangleBvhData[rightIter];
				triangleBvhData[rightIter] = tmp;
				++leftIter;
				--rightIter;
			}
		}
	}

	//TriangleBVHData is partitioned, make a BVHNode, save it to the array and recurse on the children
	++totalInnerNodes;
	BVHNode innerNode;
	innerNode.leftAABB = binInfoSAH[partitionWithMinSAH].aabbLeft;
	innerNode.rightAABB = binInfoSAH[partitionWithMinSAH].aabbRight;
	innerNode.payload.inner.leftIdx = ++allocIdx;
	innerNode.payload.inner.rightIdx = ++allocIdx;
	mBVHNodes[nodeAllocIdx] = innerNode;


	const uint32_t leftDepth = RecurseBuildBVH(triangleBvhData, startIdx, onePastLeftPartition,
		binInfoSAH[partitionWithMinSAH].centroidAABBLeft, innerNode.payload.inner.leftIdx,
		allocIdx, currentDepth+1, totalInnerNodes, totalLeafNodes);

	const uint32_t rightDepth = RecurseBuildBVH(triangleBvhData, onePastLeftPartition, onePastEndIdx, 
		binInfoSAH[partitionWithMinSAH].centroidAABBRight, innerNode.payload.inner.rightIdx, 
		allocIdx, currentDepth+1, totalInnerNodes, totalLeafNodes);

	return std::max(std::max(currentDepth, leftDepth), rightDepth);
}

uint32_t BVH::CreateLeafNode(const uint32_t startIdx, const uint32_t nodeAllocIdx,
	const uint32_t currentDepth, uint32_t& totalLeafNodes, const uint32_t numTriangles) 
{
	++totalLeafNodes;
	BVHNode node;
	node.payload.leaf.numTriangles = numTriangles;
	node.payload.leaf.numTriangles |= 0x80000000;//set msb to 1 to indicate leaf node data 
	node.payload.leaf.startIdx = startIdx;
	mBVHNodes[nodeAllocIdx] = node;
	//mBVHNodes.emplace_back(node);
	return currentDepth;
}

void BVH::SetWorldRootAABB(const glm::mat4& modelTransform) {
	const glm::vec3 localRootAABB_xdim(localRootAABB.max.x - localRootAABB.min.x, 
										0.f, 
										0.f);
	const glm::vec3 localRootAABB_zdim(0.f,
										0.f,
										localRootAABB.max.z - localRootAABB.min.z);

	//create bottom ring of 4 corners of localRootAABB
	const glm::vec4 leftBotBack = glm::vec4(localRootAABB.min, 1.f);
	const glm::vec4 rightBotBack = glm::vec4(localRootAABB.min + localRootAABB_xdim, 1.f);
	const glm::vec4 leftBotFront = glm::vec4(localRootAABB.min + localRootAABB_zdim, 1.f);
	const glm::vec4 rightBotFront = glm::vec4(localRootAABB.min + localRootAABB_xdim + localRootAABB_zdim, 1.f);

	//create top ring of 4 corners of localRootAABB
	const glm::vec4 rightTopFront = glm::vec4(localRootAABB.max, 1.f);
	const glm::vec4 rightTopBack = glm::vec4(localRootAABB.max - localRootAABB_zdim, 1.f);
	const glm::vec4 leftTopFront = glm::vec4(localRootAABB.max - localRootAABB_xdim, 1.f);
	const glm::vec4 leftTopBack = glm::vec4(localRootAABB.max - localRootAABB_xdim - localRootAABB_zdim, 1.f);


	//tranform bottom ring of corners to world
	const glm::vec4 worldleftBotBack	= modelTransform * leftBotBack;
	const glm::vec4 worldrightBotBack	= modelTransform * rightBotBack; 
	const glm::vec4 worldleftBotFront	= modelTransform * leftBotFront;
	const glm::vec4 worldrightBotFront	= modelTransform * rightBotFront;

	//tranform top ring of corners to world
	const glm::vec4 worldrightTopFront	= modelTransform * rightTopFront;
	const glm::vec4 worldrightTopBack	= modelTransform * rightTopBack; 
	const glm::vec4 worldleftTopFront	= modelTransform * leftTopFront;
	const glm::vec4 worldleftTopBack	= modelTransform * leftTopBack;

	worldRootAABB = AABB();//reset
	//can use centroid calls to build the AABB from these 8 corners
	//bot ring
	worldRootAABB.GrowAABBFromCentroid(worldleftBotBack);
	worldRootAABB.GrowAABBFromCentroid(worldrightBotBack);
	worldRootAABB.GrowAABBFromCentroid(worldleftBotFront);
	worldRootAABB.GrowAABBFromCentroid(worldrightBotFront);
	//top ring
	worldRootAABB.GrowAABBFromCentroid(worldrightTopFront);
	worldRootAABB.GrowAABBFromCentroid(worldrightTopBack);
	worldRootAABB.GrowAABBFromCentroid(worldleftTopFront);
	worldRootAABB.GrowAABBFromCentroid(worldleftTopBack);

	std::cout << "\n\n\t BVH worldRootAABB: { " << worldRootAABB.min.x << ", " << worldRootAABB.min.y << ", " << worldRootAABB.min.z << " }";
	std::cout << "  { " << worldRootAABB.max.x << ", " << worldRootAABB.max.y << ", " << worldRootAABB.max.z << " }\n";
}
