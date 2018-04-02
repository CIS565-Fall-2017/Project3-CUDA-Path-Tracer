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

#include <sstream>
#include <iostream>

BVH::BVH(const Model& model) 
	: model(model)
{
	BuildBVH();
}

struct TriAABBCentroid {
	AABB aabb;
	glm::vec3 centroid;
};

void BVH::BuildBVH() {
	/* NOTES: On fast construction of SAH-based Bounding Volume Hierarchies. Ingo Wald
		* SA (surface area) is actually the total surface area of the bounding box, not just the the face we care about
		* this paper has a node occupying half a cache line (32 bytes) storing the bounding box of the node
			some flags and the union section is either both child nodes indices or triangleID array start index and number of triangles,
			depending on whether it's an inner or leaf node.
		* when preparing to split a node, form AABB around centroids of triangles
			- There are K equidistant spaced partitions along the larges axis (checking
			 other axes doesn't seem to be worth the effort). K-1 positions along the axis to split.
			- Bins count the triangles that overlap it
			- Bins keep track of the bin bounds: the bounding box of the triangles that touch it. Could be larger or smaller than the 'domain' of the actual bin.
			- Full 3D bounds of these bind bounds are kept track of.
			- Cost Function: NumLeft*AreaLeft ++ NumRight*AreaRight
			-

		* In the setup, compute each triangles bounds and its centroid. Also compute total bounds and total centroid bounds.
			-  for SIMD friendly format store the 3D positions of the above info in four 16-byte aligned floats
			- setup stage requires 14 SSE ops per triangle. 3 loads for tri verts, two mins and maxes each for tri bounds
			 one min and max each to grow voxel bounds, one add and one mul0.5f for the centroid another min and max each for growing the centroid
			 and 3 stores to write tri bounds and centroid to memory
		
		* Triangle-to-bin projection: simply get the percentage along the centroid bounds splitting axis the triangle's centroid is
		then scale it by the number of bins times 1.f-epsilon to get the floating point bin number. Then perform float to int truncation 
		to get the bin number. 
			- initialize each bin to a negative box [+inf, -inf], allowing it to grow to include a triangles bound with one SSE min and one max
			  without having to check if empty.
			- can later merge the individual bin bounds without having to check for emptiness since growing an AABB with neg AABB using min max ops does not 
			  change the original box. 
			- Don't have to use all the tri's, just a subset but this requires param tuning to not degrade the bvh quality.
		
		* SAH evaluation: first do a linear pass from the left where you incrementally accum the bounds and num of tris for the left half. 
			storing these two values (num tris and total tris bound surface area) for each splitting plane.
			Then do the same for the right and evaluate teh SAH for each plane.
			if one of the sides of a splitting plane does not have any tris then it gets rejected as a candidate as BVH is not allowed empty partitions
		
		* In-place ID list partitioning: bvh is two contuguous arrays one for the node data one for the triangleID (ref the triangle in some way i.e. 3 vertex indices that refers to the main vertex array)
			- An in place bvh for N triangles has at most 2N-1 nodes, this can be pre-allocated
			- bvh triangles are partitioned into nodes by partioning the triangleID array (uin32_t's init to 0,1,2...N)
			- when doing sbvh will need to allocate up to spatial plit budget and create addition refs when a spatial split is neccessary.
			- can use std::partition to do the partitioning or have two iterators, one starts at teh front of the triangleID array and one at the back
			 the left iter scans to the right until it finds a tri that should be on the right of the split plane, the right iter scans to the left until
			 it finds a tri that should be on the left, then swap these two triangleID's. keep going while(iterLeft < iterRight)
			 - pass on the triangle bounds and centroid bounds of the two children to the recursive calls
		
		* Number of bins: 4-16 seems to work fine 16 being the best quality.
		This number can probably decrease as you go down the tree
		
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



	//we need to make a temporary working set of data for each triangle's AABB and centroid
	//we'll need to shuffle around the mIndices array of Model adjecent triplets of indices is a triangle that must be moved together
	std::vector<TriAABBCentroid> aabbcentroids(model.mIndices.size() / 3);

}