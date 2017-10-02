#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}
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
__global__ void kernInitLight(int numlights,
	int numGeo,
	const Geom **dev_lights,
	const Geom *dev_geoms,
	const Material *dev_materials)
{
	int lightid = 0;
	for (int i = 0; i < numGeo; ++i){
		int matId = dev_geoms[i].materialid;
		if (dev_materials[matId].emittance > 0.f)
			dev_lights[lightid++] = &dev_geoms[i];
	}
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static bool* dev_flag = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
const Geom **dev_lights=NULL;
static ShadeableIntersection * dev_intersections = NULL;
int numLights = 0;
int *dev_materialID = NULL;
unsigned *dev_pathIndices = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
void InitLight(Scene* scene) {
	const std::vector<Material>&mats = scene->materials;
	const Geom** lights = NULL;
	for (int i = 0; i < scene->geoms.size();i++)
		if (mats[scene->geoms[i].materialid].emittance > 0.f) numLights++;
	cudaMalloc(&dev_lights, numLights * sizeof(Geom *));
	kernInitLight <<<1, 1 >> >(numLights, scene->geoms.size(),dev_lights, dev_geoms, dev_materials);
	checkCUDAError("pathtraceInit");
}
void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_flag, pixelcount * sizeof(bool));
	cudaMemset(dev_flag, true, pixelcount * sizeof(bool));

	cudaMalloc(&dev_materialID, pixelcount * sizeof(unsigned));
	cudaMemset(dev_materialID, -1, pixelcount * sizeof(unsigned));

	cudaMalloc(&dev_pathIndices, pixelcount * sizeof(unsigned));

	InitLight(hst_scene);
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_flag);
	cudaFree(dev_materialID);
	cudaFree(dev_pathIndices);

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

		//segment.ray.origin = cam.position;
		//Naive
		//segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		
		//MIS
		segment.color = glm::vec3(0.f);
		segment.beta_loop = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0.f, 1.f);
		x += u01(rng);
		y += u01(rng);
		
		//depth field.
		glm::vec3 dir = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
		float ft = cam.Focus / glm::dot(cam.view, dir);
		glm::vec3 FocuspPoint = cam.position + ft * dir;
		float lensU, lensV;
		shapeSample::concentricSamplingDisk(u01(rng), u01(rng), lensU, lensV);
		glm::vec3 origin = cam.position + lensU * cam.LenRadius * cam.right + lensV * cam.LenRadius * cam.up;
		
		segment.ray.origin = origin;
		segment.ray.direction = glm::normalize(FocuspPoint - origin);

		//segment.ray.direction = glm::normalize(cam.view
		//	- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
		//	- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		//	);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

void MaterialSort(int num_paths, PathSegment *dev_paths, ShadeableIntersection *dev_intersections)
{
	thrust::device_ptr<PathSegment> dev_ptrPath(dev_paths);
	thrust::device_ptr<ShadeableIntersection> dev_ptrIntersection(dev_intersections);

	typedef thrust::tuple<thrust::device_ptr<PathSegment>, thrust::device_ptr<ShadeableIntersection>> IteratorTuple;
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	ZipIterator zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dev_ptrPath, dev_ptrIntersection));
	ZipIterator zip_end = zip_begin + num_paths;
	thrust::sort(zip_begin, zip_end, Global::MaterialCmp());
}
void MaterialIDSort(int numPaths)
{
	thrust::device_ptr<int> thrust_materialID(dev_materialID);
	thrust::device_ptr<unsigned> thrust_pathIndices(dev_pathIndices);
	thrust::stable_sort_by_key(thrust_materialID, thrust_materialID + numPaths, thrust_pathIndices);
}
// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void UpdatePathIdx(int numpaths,unsigned* dev_pathIndices) {
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < numpaths)
		dev_pathIndices[idx] = idx;
}
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, const unsigned* dev_pathIdx
	, int* dev_materialID
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	, const Material* materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths)
	{
		int path_index = dev_pathIdx[idx];
		PathSegment pathSegment = pathSegments[path_index];

		float t_min;
		glm::vec3 normal;
		int hit_geom_index = -1;
		SearchIntersection::BruteForceSearch(t_min, hit_geom_index, normal, pathSegment.ray, geoms, geoms_size);

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			int m_ID = geoms[hit_geom_index].materialid;
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = m_ID;
			intersections[path_index].surfaceNormal = normal;
			dev_materialID[idx] = m_ID;	
		}
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
void compressedPath(int& num_paths, PathSegment *paths, bool *flag)
{
	thrust::device_ptr<bool> dev_bool(flag);
	thrust::device_ptr<PathSegment> dev_ptrPaths(paths);
	thrust::remove_if(dev_ptrPaths, dev_ptrPaths + num_paths, dev_bool, thrust::logical_not<bool>());
	num_paths = thrust::count_if(dev_bool, dev_bool + num_paths, thrust::identity<bool>());
}
__global__ void shakeBSDFMaterail(
	int iter,
	int depth
	, int num_paths
	, const Geom *dev_geoms
	, int geo_size
	, const Geom **dev_lights
	,int numLights
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, unsigned* dev_pathIndices
	, bool* dev_flag
	, glm::vec3* img
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths)
	{
		int pathIndex = dev_pathIndices[idx];
		PathSegment& curPath = pathSegments[pathIndex];
		ShadeableIntersection intersection = shadeableIntersections[pathIndex];
		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathIndex, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				//curPath.color *= (materialColor * material.emittance);
				curPath.color += curPath.beta_loop*materialColor * material.emittance;
				dev_flag[pathIndex] = false;
				img[curPath.pixelIndex] += curPath.color;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				//Russian roulette
				if (depth >= 3) {
					float temp = curPath.beta_loop.x>curPath.beta_loop.y? curPath.beta_loop.x: curPath.beta_loop.y;
					float max_componenet = temp>curPath.beta_loop.z? temp: curPath.beta_loop.z;
					float q = 0.05 > (1.0f - max_componenet)? 0.05:(1.0f - max_componenet);
					float random_num = u01(rng);
					if (random_num < q)
					{
						curPath.remainingBounces = 0;;
						dev_flag[pathIndex] = false;
						img[curPath.pixelIndex] += curPath.color;
					}
					curPath.beta_loop /= (1.0f - q);
				}
				if (curPath.remainingBounces >0) {
					scatterRay(curPath,
						dev_geoms,
						geo_size,
						dev_lights,
						numLights,
						getPointOnRay(curPath.ray, intersection.t),
						intersection.surfaceNormal,
						material,
						materials,
						rng,
						u01);
					curPath.remainingBounces--;
					dev_flag[pathIndex] = curPath.remainingBounces >0 ? true : false;
					if (!dev_flag[pathIndex]) {
						img[curPath.pixelIndex] += curPath.color;
					}
				}
				
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			dev_flag[pathIndex] = false;
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
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

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

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		//cout << depth << "  " << num_paths;
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		float time = 0;
		//cudaEvent_t start, stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start);
		UpdatePathIdx << < numblocksPathSegmentTracing, blockSize1d >> >(num_paths, dev_pathIndices);
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth
			, num_paths
			, dev_paths
			, dev_pathIndices
			, dev_materialID
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_materials
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();



		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.


		//MaterialSort(num_paths, dev_paths, dev_intersections);
		MaterialIDSort(num_paths);
		shakeBSDFMaterail << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth,
			num_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_lights,
			numLights,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_pathIndices,
			dev_flag,
			dev_image);
		//shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
		//	iter,
		//	num_paths,
		//	dev_intersections,
		//	dev_paths,
		//	dev_materials
		// );
		checkCUDAError("shadeFakeMaterial");

		depth++;
		compressedPath(num_paths, dev_paths, dev_flag);


		//cudaEventRecord(stop);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&time, start, stop);

		//cout<< "   " << time << "   " << endl;
		if(!num_paths)
			iterationComplete = true; 
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
