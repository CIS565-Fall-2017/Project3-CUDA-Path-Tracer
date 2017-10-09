#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


int compactScene(int n, StreamCompaction::Efficient::CompactSupport const * aux, PathSegment * compactPath, const PathSegment * dev_path,
	    	ShadeableIntersection * compactIntersection,
	    	const ShadeableIntersection * dev_intersection);
int compactFinishedRays(int n, StreamCompaction::Efficient::CompactSupport const * aux,
	PathSegment * compactPath, const PathSegment * dev_path);
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static PathSegment * dev_paths2 = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_intersections2 = NULL;
// store the arrays and data needed to run compaction on pixel data;
static StreamCompaction::Efficient::CompactSupport aux;
static int * dev_matlTypes = NULL;
static int * dev_indices = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {

    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
  	cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections2, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_matlTypes, pixelcount *sizeof(int));
	cudaMalloc(&dev_indices, pixelcount * sizeof(int));

    // TODO: initialize any extra device memeory you need
	using StreamCompaction::Efficient::SharedScan;
	StreamCompaction::Efficient::SharedScan scan =
		  StreamCompaction::Efficient::initSharedScan(pixelcount, -1);
	aux = StreamCompaction::Efficient::initCompactSupport(scan);
	
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
	cudaFree(dev_paths2);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	cudaFree(dev_intersections2);
	cudaFree(dev_indices);
	cudaFree(dev_matlTypes);
	StreamCompaction::Efficient::freeCompaction(aux);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];
        // makeSeededRandomEngine as well.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

		segment.ray.origin = cam.position;
                segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// jitter the ray Looks like x,y (0, 0) is in
		// the top right corner
		float xpix = u01(rng);
		      xpix += static_cast<float>(x); 
			  xpix -= static_cast<float>(cam.resolution.x) * 0.5f;
			  xpix *= cam.pixelLength.x;
		float ypix = u01(rng);
		      ypix += static_cast<float>(y);
			  ypix -= static_cast<float>(cam.resolution.y) * 0.5f;
			  ypix *= cam.pixelLength.y;
       segment.ray.direction = glm::normalize(cam.view
			- cam.right * xpix
		//	- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
		    - cam.up * ypix
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// once it gets an intersection a material from the materials in the 
// intersection will be assigned.  The material may have many 
// several components and if that is the case, one component is chosen.
// Feel free to modify the code below. Now the PathSegments that miss 
// are colored black to indicate no hit.
//     There are a couple of steps taken here that are meant to optimize the time: 
//  The remaining bounces are set to zero and the color is finalized.  This occurs 
//  when the ray hits nothing or hits a light (emissive) then we give the ray its final
//  color and mark it complete.
//     This marks the pathSegments as terminated in the common cases ( hit light, hit nothing)
//  and if it hits a regular material that needs to be colored, it randomly chooses a material
//  so that the sorting works correctly in the shader.
__global__ void computeIntersections(
	int depth
	, int iter
	, int num_paths
	, PathSegment * pathSegments
	, const Geom * geoms
	, int geoms_size
	, const Material * materials
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths  && pathSegments[path_index].remainingBounces > 0)
	{
		const PathSegment& pathSegment = pathSegments[path_index];
		ShadeableIntersection& intersection = intersections[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		bool tempOutside;

		// naive parse through global geoms
			for (int i = 0; i < geoms_size; i++)
			{
				const Geom & geom = geoms[i];
				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray,
						tmp_intersect, tmp_normal, tempOutside);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray,
						tmp_intersect, tmp_normal, tempOutside);
				}
				// TODO: add more intersection tests here... triangle? metaball? CSG?

				// Compute the minimum t from the intersection tests to determine what
			  // scene geometry object was hit first.
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = i;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
					outside = tempOutside;
				}
			}
		if (hit_geom_index == -1)
		{// update the pixel color of these here too so that they get compacted away
			// here we missed the scene so the color should be black.
			// these pixels are in their final state of blackness so remainingBounces
			// is zero.
			intersection.t = -1.0f;
			intersection.matl = MaterialType::NoMaterial;
			pathSegments[path_index].color = glm::vec3(0.0f);
		        pathSegments[path_index].remainingBounces = 0;
		}
		else
		{
			//The ray hits something
			// update the intersection.
			// randomly select the material that is relevant
			// if the ray hit a light color the pathSegment.
                        thrust::default_random_engine rng = 
				makeSeededRandomEngine(iter, path_index, depth);

			intersection.t = t_min;
			int matid = geoms[hit_geom_index].materialid;
			intersection.materialId = matid;
			intersection.surfaceNormal = normal;
			const Material& m = materials[matid];
			// randomly select a material
			intersection.matl = getMaterialType(m, rng);
			intersection.outside = outside;
			if (intersection.matl == MaterialType::Emissive) {
			     ColorEmissive(pathSegments[path_index], intersection, m);
			}
		}
	}
}
// general Shader or now does lambert and reflection
__global__ void shadeMaterial(
	int iter
	, int depth
	, int num_paths
	, const ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, const Material * materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		const ShadeableIntersection& intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f  &&  pathSegments[idx].remainingBounces > 0) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			Material material = materials[intersection.materialId];
			scatterRay(pathSegments[idx], intersection, material, rng);

			// If the material indicates that the object was a light, "light" the ray
		}
		// if remaining bounces is zero of if we missed ComputeIntersections 
		// colored the light
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {


        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}
dim3 gridSize(int num_paths) 
{
	return (num_paths + StreamCompaction::Efficient::blockSize - 1) / 
		StreamCompaction::Efficient::blockSize;
}
// copy elements from unsorted intersections and pathSegments to sorted space using the pointer array.
__global__ void updateShadeableIntersections(int nPaths, const int * dev_indices, 
	         ShadeableIntersection * sortedIntersections, 
		                  const ShadeableIntersection * intersections, 
				  PathSegment * pathSegmentsSorted, const PathSegment * pathSegments)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		pathSegmentsSorted[index] = pathSegments[dev_indices[index]];
		sortedIntersections[index] = intersections[dev_indices[index]];
	}
}
// Update Material Type and indices or the pointers to the elements in the unsorted array.  Material is
// already a unique number. The pointers are initilsized to the element index (0, 1, 2, 3 ...)
__global__ void updateMaterialType(int nPaths, int * dev_matl, int * dev_indices, 
		                  const ShadeableIntersection * intersections)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		const ShadeableIntersection& si { intersections[index]};
		dev_matl[index] = si.matl << 1 + si.outside;
		dev_indices[index] = index;
	}
}
//sortByMaterial will sort the pathSegments and the ShadeableIntersections by Materialtype.
//The Shadeable Intersection has an integer MaterialType and whether it is out or in and this
//is turned into one integer and stored in dev_matl and pointers to the original locations 
//(indices 0, 1, 2 ...) in dev_indices also are stored. These are then sorted with thrust::sort_by_key
// on integer values-- and this should be very fast since radix sort can be used. I tried
// passing my own comparison function and sorting the ShadeableIntersections directly but that
// would not compile and later I understood that it would not be as fast even if it did
// since thrust would have no way to optimize the sort like it can with integers. Once that sort is done,
//I go back and move the Shadeable  intersections and pathSegments.
void sortByMaterial(int nPaths,  dim3 blocks, int * dev_matl,
		int * dev_indices, 
		ShadeableIntersection * sortedIntersections, 
		 const ShadeableIntersection * intersections, 
	         PathSegment * pathSegmentsSorted, 
		 const PathSegment * pathSegments)
{
	dim3 grid { gridSize(nPaths)};
	updateMaterialType<<<grid, blocks>>>(nPaths, dev_matl,
			 dev_indices, intersections);
	thrust::device_ptr<int> matl(dev_matl);
	thrust::device_ptr<int> loc(dev_indices);
	thrust::sort_by_key(matl, matl + nPaths, loc);
//	thrust::device_ptr<ShadeableIntersection> intTest(intersections);
//	thrust::sort_by_key(intTest, intTest + 5, intTest, thrust::less<ShadeableIntersection>() );
    updateShadeableIntersections<<<grid, blocks>>>(nPaths, 
		           dev_indices,  sortedIntersections, intersections, pathSegmentsSorted, pathSegments);
}
// addTerminated rays
// all rays with remainingBounces == 0 will get added to the image with no sorting.
__global__ void addTerminatedRays(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if ( iterationPath.remainingBounces == 0) {
		        image[iterationPath.pixelIndex] += iterationPath.color;
			// iterationPaths[index].color = glm::vec3(0.0f); to get this
			// to work with no compaction
		}
	}
}
// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
	//const int traceDepth = 1;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	dim3 blockSize1d = StreamCompaction::Efficient::blockSize;
	int depth = 0;
	int num_paths = pixelcount;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	// clean shading chunks
	//cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
	// tracing
	dim3 numblocksPathSegmentTracing =gridSize(num_paths);
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, iter
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_materials
		, dev_intersections
		);
	checkCUDAError("traceComputeIntersections");
	cudaDeviceSynchronize();
	// compact the threads to remove paths that are zero.  This call puts all the 
	// active threads in the beginning; 
	if (hst_scene->state.method != pathTraceMethod::NoCompaction){
		// adds terminated rays to the image
	    addTerminatedRays<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, 
			    dev_image, dev_paths);
	    int newpaths { compactScene(num_paths, &aux, dev_paths2, dev_paths, dev_intersections2,
		         dev_intersections)};
	    num_paths = (newpaths);
		if (hst_scene->state.method == pathTraceMethod::CompactionWithSorting) {
			sortByMaterial(num_paths, blockSize1d, dev_matlTypes, dev_indices,
				dev_intersections, dev_intersections2, dev_paths, dev_paths2);
		}
		else {
	           std::swap(dev_intersections2, dev_intersections); 
	           std::swap(dev_paths, dev_paths2);  
		}
	
            // all the new terminated rays (some may be light colored and some are black) 
            // need to be placed in the back of dev_paths2
           // int newZeros { compactFinishedRays(num_paths, &aux, dev_paths2 + newpaths, dev_paths)};
           // if (newZeros + newpaths != num_paths) {
            //	throw std::runtime_error("Partitioning of PathSegments"
            //			"does not account for all pixels");
            //}
            // send those data to the image
            //dim3 finishedblocks = gridSize(newZeros);
            //finalGather<<<finishedblocks, blockSize1d>>>(newZeros, dev_image, dev_paths2 + newpaths);
	    // dev_paths2 is the one organized with pixels that need to be shaded--
	    // Use swap to set the organized buffer to dev_paths
	}
	// dev_paths2 is the one organized with pixels that need to be shaded--
	// Use swap to set the organized buffer to dev_paths
	checkCUDAError("CompactScene");


  /*shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
	  );*/
        dim3 numBlocksPathShading = gridSize(num_paths);   
        shadeMaterial<<<numBlocksPathShading, blockSize1d>>> (
          iter,
          depth,
          num_paths,
          dev_intersections,
          dev_paths,
          dev_materials);
        checkCUDAError("Shade Material");
        depth++;
        iterationComplete = (depth == traceDepth) || num_paths == 0; // TODO: should be based off stream compaction results.
     }

  // Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = gridSize(num_paths);
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
//MapInActivePixels will update the bool array with a true for 
// all active pixel threads with remaining Bounces
// N is the size of the bool array and should be a power of 2
// n is the size of the elements in the pathSegments array  n must be <= N.
__global__ void kernMapInActivePixels(int N, int n, int * dev_booldata, const
		PathSegment * pathSegments)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= N) {
              return;
        }
        dev_booldata[index] = ( index < n && pathSegments[index].remainingBounces == 0) ?
		            1 : 0;
}
//MapActivePixels will update the bool array with a true for 
// all active pixel threads with remaining Bounces
// N is the size of the bool array and should be a power of 2
// n is the size of the pathSegments array  n must be <= N.
__global__ void kernMapActivePixels(int N, int n, int * dev_booldata, const
		PathSegment * pathSegments)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= N) {
              return;
        }
        dev_booldata[index] = ( index < n && pathSegments[index].remainingBounces > 0) ?
		            1 : 0;
}
/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatterPixelData(int n, PathSegment *compactPath,
	 const PathSegment * dev_path, ShadeableIntersection * compactIntersection,
         const ShadeableIntersection * dev_intersection, const int *bools, const int *indices) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n) {
		return;
	}
        if ( bools[index] == 1) {
		compactPath[indices[index]]      = dev_path[index];
		compactIntersection[indices[index]] = dev_intersection[index];
	}
}
/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatterSegments(int n, PathSegment *compactPath,
	 const PathSegment * dev_path, const int *bools, const int *indices) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n) {
		return;
	}
        if ( bools[index] == 1) {
		compactPath[indices[index]]      = dev_path[index];
	}
}

 // n     size of dev_intersection, dev_path
 // aux   pointers to scan data arrays and bool array
 int compactScene(int n, StreamCompaction::Efficient::CompactSupport const * aux, 
  	       PathSegment * compactPath, const PathSegment * dev_path,
                 ShadeableIntersection * compactIntersection,
                 const ShadeableIntersection * dev_intersection)
 {
  	
          int logn { ilog2ceil(n)};
  	if ( logn > aux -> scan.log2N) {
  		throw std::runtime_error("Input Length greater than"
  				  "  that allocated in Scan Data");
  	}
          int N{ 1 << logn };

  	dim3  fullBlocksPerGrid((N + StreamCompaction::Efficient::blockSize - 1) / 
		                         StreamCompaction::Efficient::blockSize);
          // initialize the bool Data with the active pixels
  	kernMapActivePixels << <fullBlocksPerGrid,
  		StreamCompaction::Efficient::blockSize >> >(N, n, aux -> bool_data, dev_path);
  	// copy bools over to indices
  	cudaMemcpy(aux -> scan.dev_idata, aux -> bool_data, N * sizeof(int),
  		cudaMemcpyDeviceToDevice);
  	// do the shared scan of the indices
  	StreamCompaction::Efficient::efficientScanShared(logn, aux -> scan.log2T, aux -> scan.dev_idata, 
  			    aux -> scan.scan_sum);
  	kernScatterPixelData<<< fullBlocksPerGrid, 
		StreamCompaction::Efficient::blockSize>>>(n, compactPath, dev_path, compactIntersection, 
  			    dev_intersection, aux -> bool_data, aux -> scan.dev_idata);
  	int  lastIndex;
	StreamCompaction::Efficient::transferIntToHost(1, &lastIndex, aux -> scan.dev_idata + N - 1);
  	int lastIncluded;
	StreamCompaction::Efficient::transferIntToHost(1, &lastIncluded, aux -> bool_data + N - 1);
        lastIndex += lastIncluded;	
        return lastIndex;
 }
 // n     size of dev_intersection, dev_path
 // aux   pointers to scan data arrays and bool array
 int compactFinishedRays(int n, StreamCompaction::Efficient::CompactSupport const * aux, 
  	       PathSegment * compactPath, const PathSegment * dev_path)
 {
  	
        int logn { ilog2ceil(n)};
  	if ( logn > aux -> scan.log2N) {
  		throw std::runtime_error("Input Length greater than"
  				  "  that allocated in Scan Data");
  	}
          int N{ 1 << logn };

  	dim3  fullBlocksPerGrid((N + StreamCompaction::Efficient::blockSize - 1) / 
		                         StreamCompaction::Efficient::blockSize);
          // initialize the bool Data with the active pixels
  	kernMapInActivePixels << <fullBlocksPerGrid,
  		StreamCompaction::Efficient::blockSize >> >(N, n, aux -> bool_data, dev_path);
  	// copy bools over to indices
  	cudaMemcpy(aux -> scan.dev_idata, aux -> bool_data, N * sizeof(int),
  		cudaMemcpyDeviceToDevice);
  	// do the shared scan of the indices
  	StreamCompaction::Efficient::efficientScanShared(logn, aux -> scan.log2T, aux -> scan.dev_idata, 
  			    aux -> scan.scan_sum);
  	kernScatterSegments<<< fullBlocksPerGrid, 
		StreamCompaction::Efficient::blockSize>>>(n, compactPath, dev_path,  aux -> bool_data,
				aux -> scan.dev_idata);
  	int  lastIndex;
	StreamCompaction::Efficient::transferIntToHost(1, &lastIndex, aux -> scan.dev_idata + N - 1);
  	int lastIncluded;
	StreamCompaction::Efficient::transferIntToHost(1, &lastIncluded, aux -> bool_data + N - 1);
        lastIndex += lastIncluded;	
        return lastIndex;
 }

	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.
    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

