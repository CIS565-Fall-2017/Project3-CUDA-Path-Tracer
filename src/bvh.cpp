#include "bvh.h"
#include <algorithm>

int MaxPrimsInNode;



BVHBuildNode *recursiveBuild(std::vector<BVHPrimitiveInfo> &primitiveInfo,
	int start, int end, int& totalNodes,
	std::vector<Triangle> &orderedPrims,
	std::vector<Triangle> &primitives) {

	BVHBuildNode *node = new BVHBuildNode();

	totalNodes++;

	// Compute bounds of all primitives in BVH node
	Bounds3f bounds = primitiveInfo[start].bounds;

	for (int i = start; i < end; i++) {
		bounds = Union(bounds, primitiveInfo[i].bounds);
	}

	int nPrimitives = end - start;

	if (nPrimitives == 1) {
		// Create leaf
		int firstPrimOffset = orderedPrims.size();
		for (int i = start; i < end; i++) {
			int primNum = primitiveInfo[i].primitiveNumber;
			orderedPrims.push_back(primitives[primNum]);
		}
		node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
		return node;
	}

	else {
		// Compute bound of primitive centroids, choose split dimension dim
		Bounds3f centroidBounds = Bounds3f(primitiveInfo[start].centroid);
		for (int i = start; i< end; i++) {
			centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid);
		}
		int dim = centroidBounds.MaximumExtent();


		// Partition primitives into two set and build children
		float mid = ((float)start + (float)end) / 2.0f;
		if (centroidBounds.max[dim] == centroidBounds.min[dim]) {
			// Create leaf BVHBuildNode
			int firstPrimOffset = orderedPrims.size();
			for (int i = start; i < end; i++) {
				int primNum = primitiveInfo[i].primitiveNumber;
				orderedPrims.push_back(primitives[primNum]);
			}

			node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
			return node;
		}
		else {
			// ------------------------------------------
			// Partition primitives using approximate SAH
			// if nPrimitives is smaller than 4,
			// we partition them into equally sized subsets
			// ------------------------------------------

			if (nPrimitives <= 2) {
				//Partition primitives into equally sized subsets
				//mid = (start + end) / 2;
				mid = ((float)start + (float)end) / 2.0f;
				std::nth_element(&primitiveInfo[start], &primitiveInfo[(int)mid],
					&primitiveInfo[end - 1] + 1,
					[dim](const BVHPrimitiveInfo &a, const BVHPrimitiveInfo &b) {
					return a.centroid[dim] < b.centroid[dim];
				});
			}

			else {
				// Allocate BucketInfo for SAH partition buckets
				constexpr int nBuckets = 12;

				BucketInfo buckets[nBuckets];

				// Initialize BucketInfo for SAH partition buckets
				for (int i = start; i < end; i++) {
					int b = nBuckets * centroidBounds.Offset(primitiveInfo[i].centroid)[dim];
					if (b == nBuckets) b = nBuckets - 1;
					buckets[b].count++;
					buckets[b].bounds = Union(buckets[b].bounds, primitiveInfo[i].bounds);
				}

				// Compute costs for splitting after each bucket
				float cost[nBuckets - 1];
				for (int i = 0; i < nBuckets - 1; i++) {
					Bounds3f b0;
					Bounds3f b1;
					//                    Bounds3f b0, b1;
					int count0 = 0;
					int count1 = 0;
					for (int j = 0; j <= i; j++) {
						b0 = Union(b0, buckets[j].bounds);
						count0 += buckets[j].count;
					}
					for (int j = i + 1; j < nBuckets; j++) {
						b1 = Union(b1, buckets[j].bounds);
						count1 += buckets[j].count;
					}
					cost[i] = 1.f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) / bounds.SurfaceArea();
				}

				// Find bucket to split at that minimizes SAH metric
				float minCost = cost[0];
				int minCostSplitBucket = 0;
				for (int i = 1; i < nBuckets - 1; i++) {
					if (cost[i] < minCost) {
						minCost = cost[i];
						minCostSplitBucket = i;
					}
				}

				// Either create leaf or split primitives at selected SAH bucket
				float leafCost = nPrimitives;
				if (nPrimitives > MaxPrimsInNode || minCost < leafCost) {
					BVHPrimitiveInfo *pmid = std::partition(&primitiveInfo[start],
						&primitiveInfo[end - 1] + 1,
						[=](const BVHPrimitiveInfo &pi) {
						int b = nBuckets * centroidBounds.Offset(pi.centroid)[dim];
						if (b == nBuckets) b = nBuckets - 1;
						return b <= minCostSplitBucket;
					});
					mid = pmid - &primitiveInfo[0];
				}
				else {
					// Create leaf BVHBuildNode
					int firstPrimOffset = orderedPrims.size();
					for (int i = start; i < end; i++) {
						int primNum = primitiveInfo[i].primitiveNumber;
						orderedPrims.push_back(primitives[primNum]);
					}

					node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
					return node;
				}
			}
			// ------------------------------------------
			// --------- partition part end -------------
			// ------------------------------------------

			node->InitInterior(dim,
				recursiveBuild(primitiveInfo, start, (int)mid, totalNodes, orderedPrims, primitives),
				recursiveBuild(primitiveInfo, (int)mid, end, totalNodes, orderedPrims, primitives));
		}

	}
	return node;
}


int flattenBVHTree(BVHBuildNode *node, int *offset, LinearBVHNode *bvh_nodes) {

	LinearBVHNode *linearNode = &bvh_nodes[*offset];
	linearNode->bounds = node->bounds;
	int myOffset = (*offset)++;
	if (node->nPrimitives > 0) {
		linearNode->primitivesOffset = node->firstPrimOffset;
		linearNode->nPrimitives = node->nPrimitives;
	}
	else {
		// Create interior flattened BVH node
		linearNode->axis = node->splitAxis;
		linearNode->nPrimitives = 0;
		flattenBVHTree(node->children[0], offset, bvh_nodes);
		linearNode->secondChildOffset = flattenBVHTree(node->children[1], offset, bvh_nodes);
	}

	return myOffset;
}

void deleteBuildNode(BVHBuildNode *root) {

	// If this is a leaf node
	if (root->children[0] == nullptr &&
		root->children[1] == nullptr) {
		delete root;
		return;
	}

	deleteBuildNode(root->children[0]);

	deleteBuildNode(root->children[1]);

	delete root;

	return;
}

// Constructs an array of BVHPrimitiveInfos, recursively builds a node-based BVH
// from the information, then optimizes the memory of the BVH
LinearBVHNode* ConstructBVHAccel(int& totalNodes, std::vector<Triangle> &primitives, int maxPrimsInNode)
{

	MaxPrimsInNode = glm::min(255, maxPrimsInNode);

	// Primitives are only Triangles NOW


	// Test whether primitives is empty or not
	int primitivesSize = primitives.size();

	if(primitivesSize == 0){
        return nullptr;
    }


    // Build BVH from primitives
    // 1. Initialize permitiveInfo array for primitives
    std::vector<BVHPrimitiveInfo> primitiveInfo(primitives.size());
    for(size_t i = 0; i < primitivesSize; i++){
        primitiveInfo[i] = BVHPrimitiveInfo(i, primitives[i].WorldBound());
    }



    // 2. Build BVH tree for primitives using primitiveInfo
    totalNodes = 0;
    std::vector<Triangle> orderedPrims;
    orderedPrims.reserve(primitivesSize);
    BVHBuildNode *root;

    root = recursiveBuild(primitiveInfo, 0, primitivesSize, totalNodes, orderedPrims, primitives);

    primitives.swap(orderedPrims);


    // 3. Compute representation of depth-first traversal of BVH tree
	LinearBVHNode *bvh_nodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    flattenBVHTree(root, &offset, bvh_nodes);


    // 4. delete BVHBuildNode root
    deleteBuildNode(root);

	return bvh_nodes;
}


void DeconstructBVHAccel(LinearBVHNode *bvh_nodes)
{
	delete[] bvh_nodes;
}




