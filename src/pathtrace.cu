#pragma once
#include <cstdio>
#include <cuda.h>
#include <curand.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/component_wise.hpp"
#include "utilities.h"
#include "utilkern.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "stream_compaction\sharedandbank.h" 
#include "stream_compaction\radix.h" 

#define ERRORCHECK			1
#define COMPACT				0 //0 NONE, 1 THRUST, 2 CUSTOM(breaks render, just use for timing compare)
#define PT_TECHNIQUE		1 //0 NAIVE, 1 MIS 
#define FIRSTBOUNCECACHING	0
#define TIMER				0
#define MATERIALSORTING		0 
//https://thrust.1ithub.io/doc/group__stream__compaction.html#ga5fa8f86717696de88ab484410b43829b
//https://stackoverflow.com/questions/34103410/glmvec3-and-epsilon-comparison
struct isDead { //needed for thrust's predicate, the last arg in remove_if
	__host__ __device__
		bool operator()(const PathSegment& path) { //keep the true cases
		return (path.remainingBounces > 0);
	}
};

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
	std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}
//ALREADY DEFINED IN STREAM_COMPACTION
//#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
//#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
//void checkCUDAErrorFn(const char *msg, const char *file, int line) {
//#if ERRORCHECK
//    cudaDeviceSynchronize();
//    cudaError_t err = cudaGetLastError();
//    if (cudaSuccess == err) {
//        return;
//    }
//
//    fprintf(stderr, "CUDA error");
//    if (file) {
//        fprintf(stderr, " (%s:%d)", file, line);
//    }
//    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
//#  ifdef _WIN32
//    getchar();
//#  endif
//    exit(EXIT_FAILURE);
//#endif
//}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
	//int x = resolution.x-1 - (blockIdx.x*blockDim.x + threadIdx.x);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

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
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
//need something for lights
static int numlights = 0;
static int* dev_geomLightIndices = NULL;
static ShadeableIntersection * dev_firstbounce = NULL;
static int* dev_materialIDsForPathsSort = NULL;
static int* dev_materialIDsForIntersectionsSort = NULL;

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

  	cudaMalloc(&dev_firstbounce, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_firstbounce, 0, pixelcount * sizeof(ShadeableIntersection));
    // TODO: initialize any extra device memeory you need
	//check which geoms are emissive and copy them to its own array(for large scenes 
	//rather not pass geoms if we don't have to)
	std::vector<int> geomLightIndices;
	for (int i = 0; i < scene->geoms.size(); ++i) {
		int materialID = scene->geoms[i].materialid;
		if (scene->materials[materialID].emittance > 0) {
			geomLightIndices.push_back(i);
			numlights++;
		}
	}
	cudaMalloc((void**)&dev_geomLightIndices, numlights * sizeof(int));
  	cudaMemcpy(dev_geomLightIndices, geomLightIndices.data(), numlights * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_materialIDsForPathsSort, pixelcount * sizeof(int));
	cudaMalloc((void**)&dev_materialIDsForIntersectionsSort, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_geomLightIndices);
	numlights = 0;
  	cudaFree(dev_firstbounce);
	cudaFree(dev_materialIDsForPathsSort);
	cudaFree(dev_materialIDsForIntersectionsSort);

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
__global__ void generateRayFromCamera(const Camera cam, const int iter, const int traceDepth, PathSegment* pathSegments) {
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) return;

	int index = x + (y * cam.resolution.x);
	PathSegment& segment = pathSegments[index];

	segment.ray.origin = cam.position;

	// TODO: implement antialiasing by jittering the ray
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u(-0.5f, 0.5f);

	//camera top right is (0,0)
#if 1 == FIRSTBOUNCECACHING
	segment.ray.direction = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * (x - cam.resolution.x * 0.5f)
		- cam.up    * cam.pixelLength.y * (y - cam.resolution.y * 0.5f) );
#else
	segment.ray.direction = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * (x - cam.resolution.x * 0.5f + u(rng))
		- cam.up    * cam.pixelLength.y * (y - cam.resolution.y * 0.5f + u(rng)) );
#endif

	segment.MSPaintPixel = glm::ivec2(cam.resolution.x - 1 - x, y);
	segment.pixelIndex = index;
	segment.remainingBounces = traceDepth;
	segment.color = glm::vec3(1.f);

#if 1 == PT_TECHNIQUE //MIS
	segment.color = glm::vec3(0.f);
	segment.throughput = glm::vec3(1.f);
	segment.specularbounce = false;
#endif
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(const int iter, const int depth,
	const int num_paths, PathSegment * pathSegments,
	const Geom* const geoms, const int geoms_size,
	ShadeableIntersection* const intersections,
	ShadeableIntersection* const firstbounce, const int firstbouncecaching) 
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int actual_first_iter = 1;//iter 0 is an initialization step

	if (path_index >= num_paths) { return; }
	PathSegment pathSegment = pathSegments[path_index];

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;//why is this needed if the normal is being corrected already

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	//FIRSTBOUNCECACHING
	// naive parse through global geoms
	if ((1 == firstbouncecaching && 0 == depth && actual_first_iter == iter) || (0 < depth) || (0 == firstbouncecaching)) {
		for (int i = 0; i < geoms_size; i++) {
			const Geom & geom = geoms[i];
			t = shapeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

			//min
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}//for i < geoms_size
	}//if we need to find closest intersection

	if (1 == firstbouncecaching && 0 == depth && actual_first_iter == iter) {//first time we call computeintersection
		if (-1 == hit_geom_index) {
			firstbounce[path_index].t = -1.f;
			intersections[path_index].t = -1.f;
		} else {
			firstbounce[path_index].t = t_min;
			firstbounce[path_index].materialId = geoms[hit_geom_index].materialid;
			firstbounce[path_index].surfaceNormal = normal;
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	} else if (0 == depth && actual_first_iter < iter && 1 == firstbouncecaching) { //first depth in other iters when firstbounce is enabled
		intersections[path_index].t = firstbounce[path_index].t;
		intersections[path_index].materialId = firstbounce[path_index].materialId;
		intersections[path_index].surfaceNormal = firstbounce[path_index].surfaceNormal;
	} else {//first bounce is off or depth is greater than 0, do it normally
		if (-1 == hit_geom_index) {
			intersections[path_index].t = -1.f;
		} else {
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}
////////////////////////////////////////////////////
////////////           NAIVE            ////////////
////////////////////////////////////////////////////
__global__ void shadeMaterialNaive (
  const int iter, const int num_paths, 
	const ShadeableIntersection * shadeableIntersections, 
	PathSegment * pathSegments, const Material * materials
	)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_paths) { return; }

	PathSegment& path = pathSegments[idx];
	const ShadeableIntersection& isect = shadeableIntersections[idx];

#if 0 == COMPACT
	if (0 >= path.remainingBounces) { return; }
#endif

	//Hit Nothing
	if (0.f >= isect.t) {// Lots of renderers use 4 channel color, RGBA, where A = alpha, often // used for opacity, in which case they can indicate "no opacity".  // This can be useful for post-processing and image compositing.
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
		return;
	}

	const Material& material = materials[isect.materialId];

	//Hit Light
	if (0.f < material.emittance) {
		path.color *= (material.color * material.emittance); 
		path.remainingBounces = 0;
		return;
	}

	//this was last chance to hit light
	if (0 >= --path.remainingBounces) {
		path.color = glm::vec3(0.f);
		path.remainingBounces = 0;
		return;
	}

	//Hit Material, generate new ray for the path(wi), get pdf and color for the material, use those to mix with the path's existing color
	float bxdfPDF; glm::vec3 bxdfColor;
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
	scatterRayNaive(path, isect, material, bxdfPDF, bxdfColor, rng);
	if (0 >= bxdfPDF || isBlack(bxdfColor)) {//for total internal reflection
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
		return;
	}
	//if (bxdfPDF > 0.f) { bxdfColor /= bxdfPDF; }
	bxdfColor /= bxdfPDF;
	path.color *= bxdfColor * absDot(isect.surfaceNormal, path.ray.direction);

}

///////////////////////////////////////////////////
////////////MULTIPLE IMPORTANCE SAMPLING///////////
///////////////////////////////////////////////////
__global__ void shadeMaterialMIS(const int iter, const int depth,
	const int numlights, const int MAXBOUNCES,
	const int num_paths, const ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments, const Material* materials, 
	const int* dev_geomLightIndices, const Geom* dev_geoms, const int numgeoms) 
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int terminatenumber = -100;

	if (idx >= num_paths) { return; }
	PathSegment& path = pathSegments[idx];
	const ShadeableIntersection& isect = shadeableIntersections[idx];
	
	/////////////
	////DEBUG////
	/////////////
	//if (idx == 0) { 
		//printf("\nhello %i", idx); //you can printf from the gpu!
	//}
	//int pixelx = path.MSPaintPixel.x;
	//int pixely = path.MSPaintPixel.y;
	//if (760 == pixelx && 310 == pixely) {
	//	printf("\npixelx: %i, pixely: %i, depth: %i, iter: %i", pixelx, pixely, depth, iter);
	//}




#if 0 == COMPACT
	if (terminatenumber >= path.remainingBounces) { return; }
#endif

	//Hit Nothing
	if (0.f >= isect.t) {
		path.remainingBounces = terminatenumber;
		return;
	}

	const Material& m = materials[isect.materialId];

	//First Bounce or Last bounce was specular
	//NOTE: may want to remove spec bounce check as the matchesflags from 561 didnt work
	const glm::vec3 wo = -path.ray.direction;
	if (0 == depth || path.specularbounce) { //what if path color started at 1 and you multiplied as you went along???
		path.color += path.throughput * Le(m, isect.surfaceNormal, wo); 
	}

	//Hit Light
	if (0.f < m.emittance) {
		path.remainingBounces = terminatenumber;//already accounted for Le above
		return;
		//also has something where if a bxdf was not produced
		//(lights just return) then set the origin at the intersect point 
		//and continue on in same direction and path.remainingBounces++?
	}

	//if(material doesnt exist) { path.remainingBounces = terminatenumber; return; }

	path.specularbounce = (m.hasReflective || m.hasRefractive) ? 1 : 0;

	//////////////////////
	//////  DIRECT  //////
	//////////////////////
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);
	float pdfdirect = 0;
	glm::vec3 widirect(0.f);
	glm::vec3 colordirect(0.f);

	const int randlightindex = dev_geomLightIndices[int( u01(rng)*numlights )];
	const Geom& randlight = dev_geoms[randlightindex];
	const Material& mlight = materials[randlight.materialid];
	const glm::vec3 pisect = getPointOnRay(path.ray, isect.t) + isect.surfaceNormal*EPSILON;

	////////////////////////
	/////// LIGTH SAMPLE////
	////////////////////////
	//NOTE: COLORDIRECT WILL BE 0 FOR PURE SPECULAR STUFF
	//BECUASE CHANCES OF THE REFLECTED VIEW RAY LINING UP WITH OUR
	//SAMPLED LIGHT DIRECTION IS ZERO
	//BUT COLOR DIRECT SAMPLE SHOULD RETURN SOMETHING IF THE REFLECTED
	//RAY (OUR BXDF SAMPLE WI) HITS A LIGHT
	//SO MAYBE IF SPECULAR BOUNCE, SKIP THE LIGHT SAMPLING PART AND BUT NOT THE BXDF SAMPLEING 
	//FOR HITTING THE LIGHT PART

	//int pixelx = path.MSPaintPixel.x;
	//int pixely = path.MSPaintPixel.y;
	//if (143 == pixelx && 557 == pixely && 0 == depth) {
	//	printf("\npixelx: %i, pixely: %i, depth: %i, iter: %i", pixelx, pixely, depth, iter);
	//}
	//sample the light to get widirect, pdfdirect, and colordirect
	//you'll want to switch the cube light to a planar light to make things easier?
	colordirect = sampleLight(randlight, mlight, pisect, 
		isect.surfaceNormal, rng, widirect, pdfdirect);
	pdfdirect /= numlights;

	//spawn ray from isect point and use widirect for direction 
	Ray wiray_direct;
	wiray_direct.origin = pisect;
	wiray_direct.direction = widirect;

	//get the closest intersection in scene
	int hitgeomindex = -1;
	findClosestIntersection(wiray_direct, dev_geoms, numgeoms, hitgeomindex);

	//TODO: prob only need first and last condition
	if (0 >= pdfdirect || 0 > hitgeomindex || hitgeomindex != randlightindex) {
		colordirect = glm::vec3(0);
	} else if (0.f < pdfdirect) {//condition needed? is OR async on gpu?
		glm::vec3 bxdfcolordirect; float bxdfpdfdirect;
		bxdf_FandPDF(wo, widirect, isect.surfaceNormal, 
			m, bxdfcolordirect, bxdfpdfdirect); 
		const float absdotdirect = absDot(widirect, isect.surfaceNormal);
		const float powerheuristic_direct = powerHeuristic(1,pdfdirect,1,bxdfpdfdirect);
		colordirect = (bxdfcolordirect * colordirect * absdotdirect * powerheuristic_direct) / pdfdirect;
	}
	////^^^^^^^^^^^^TODO: Turn into function^^^^^^^^^^^^^^^^^^^^^

	///////////////////////////////////////////////////////////////
	////////// SAMPLE BXDF, CHECK IF HIT A LIGHT //////////////////
	////////// PROBABLY NEED TO MAKE SURE YOU'RE HITTING A DIFFERNET LIGHT THAN THE ONE WE RANDOMLY SAMPLED////////
	////////// SO THAT WE DON'T OVER SAMPLE THE LIGHT////////
	////////// USE GETCLOSESTINTERSECTION TO MAKE SURE ITS A DIFFERENT LIGHT////////
	///////////////////////////////////////////////////////////////
	glm::vec3 colordirectsample(0);
	glm::vec3 widirectsample;
	float pdfdirectsample;
	PathSegment pathcopy = path;//scatterRayNaive updates the path with isect origin and wi direction
	scatterRayNaive(pathcopy, isect, m, pdfdirectsample, colordirectsample, rng);
	widirectsample = pathcopy.ray.direction;
	if (glm::length2(colordirectsample) > 0 && pdfdirectsample > 0 && isBlack(colordirect)) {//only take if we got nothing for surface sampling part
		float DLPdf_for_directsample = lightPdfLi(randlight, pisect, widirectsample);

		if (DLPdf_for_directsample > 0) {//widirectsample can hit it (might not be closest though)
			Ray widirectsampleRay; widirectsampleRay.direction = widirectsample; widirectsampleRay.origin = pisect;
			int posslightindex;//we know a our light is in the widirectsample direction, lets try to hit it, and hopefully its the closest light
			const glm::vec3 nposslight = findClosestIntersection(widirectsampleRay, dev_geoms, numgeoms, posslightindex);

			if (posslightindex == randlightindex) {//only want one light per direct lighting sample pair(need two to cover edge cases of combos of light size/intensity and bxdf)
				DLPdf_for_directsample /= numlights;
				const float powerheuristic_directsample = powerHeuristic(1, pdfdirectsample, 1, DLPdf_for_directsample);
				const float absdotdirectsample = absDot(widirectsample, isect.surfaceNormal);
				const Geom& posslight = dev_geoms[posslightindex];
				const Material& mposslight = materials[posslight.materialid];
				const glm::vec3 Li = Le(mposslight, nposslight, -widirectsample);//returns 0 if we don't hit a light(no emmisive)
				colordirectsample = (colordirectsample * Li * absdotdirectsample * powerheuristic_directsample) / pdfdirectsample;
			} else {
				colordirectsample = glm::vec3(0);
			}
		} else {
			colordirectsample = glm::vec3(0);
		}
	} else {//prob dont need 
		colordirectsample = glm::vec3(0);
	}
	///////^^^^^^^^^^^TODO: Turn into function^^^^^^^^^^^^^^^^^^

	//IF SPECULARBOUNCE, PROB SHOULDNT DO ANY OF THE DIRECT LIGHTING SINCE WE WILL DOUBLE COUNT IT IF IT WAS HEADING TOWARDS A LIGHT ANYWAY
	colordirect = path.specularbounce ? glm::vec3(0) : colordirect + colordirectsample;
	//colordirect = colordirect + colordirectsample;
	
	path.color += path.throughput*colordirect;

	////////////////////////
	//////  INDIRECT  //////
	////////////////////////
	//get global illum ray from bxdf and loop
	//do a normal samplef to get wi pdf and color
	float bxdfPDF; glm::vec3 bxdfColor;
	scatterRayNaive(path, isect, m, bxdfPDF, bxdfColor, rng);
	const glm::vec3 bxdfWi = path.ray.direction;
	const glm::vec3 normal = isect.surfaceNormal;

	glm::vec3 current_throughput(0.f); 
	if(!isBlack(bxdfColor) && 0 < bxdfPDF) { 
		current_throughput = bxdfColor * absDot(bxdfWi, normal) / bxdfPDF;
	}
	path.throughput *= current_throughput;
	
	//russian roulette terminate low-energy rays
	if(depth>MAXBOUNCES) {
		const float max = glm::compMax(path.throughput);
		if (max < u01(rng)) {
			path.remainingBounces = terminatenumber;
			return;
		}
		//in the off chance this ray is still going after a long time
		//scale it up to reduce noise 
		//i.e. make it more like the rays in its pixel who terminated earlier
		//this one just got lucky presumably
		path.throughput /= max;
	}

}
// Add the current iteration's output to the overall image
__global__ void finalGather(const int nPaths, const glm::ivec2 resolution, 
	glm::vec3 * image, const PathSegment * iterationPaths)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nPaths) { return; }

	PathSegment iterationPath = iterationPaths[index];
	image[iterationPath.pixelIndex] += iterationPath.color;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) { //, const int MAXBOUNCES) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);//8 by 8 pixels
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

	//dev_path holds them all (they are calculated for world space)
	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	

#if 1 == TIMER
	using time_point_t = std::chrono::high_resolution_clock::time_point;
	time_point_t time_start_cpu = std::chrono::high_resolution_clock::now();
#endif
	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks, prob only needs to be num_paths not pixelcount
		//cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		//checkCUDAError("cuda memset dev_intersections");

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), 
			dev_intersections, dev_firstbounce, FIRSTBOUNCECACHING
			);
		checkCUDAError("trace one bounce");
		//cudaDeviceSynchronize();
#if 1 == MATERIALSORTING
	//copy material ids to to two dev material id arrays. 1 needed for sorting paths and 1 for isects.
		//our threadid for shading is used to index into dev_intersections and dev_paths
		copyMaterialIDsToArrays << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths, dev_materialIDsForIntersectionsSort, dev_materialIDsForPathsSort, dev_intersections);
	//thrust sort sorts the second array by how it sorted the first array.
		thrust::sort_by_key(thrust::device, dev_materialIDsForIntersectionsSort, dev_materialIDsForIntersectionsSort + num_paths, dev_intersections);
		thrust::sort_by_key(thrust::device, dev_materialIDsForPathsSort, dev_materialIDsForPathsSort + num_paths, dev_paths);
#endif


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if 0 == PT_TECHNIQUE
		shadeMaterialNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, 
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		checkCUDAError("shadeMaterial");
#elif 1 == PT_TECHNIQUE
		shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth, numlights, traceDepth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials, dev_geomLightIndices, dev_geoms, hst_scene->geoms.size()
			);
		checkCUDAError("shadeMaterial");
#else
		printf("\n unknown PT_TECHNIQUE \n");
#endif

		depth++;

		//Do compact and determine how many paths are still left, our compact only operates on ints...so use thrust
#if 0 == COMPACT
		if (traceDepth == depth) { iterationComplete = true; num_paths = 0; }
#elif 1 == COMPACT
		//PathSegment *compactend = thrust::remove_if(dev_paths, dev_paths + num_paths, isDead());
		PathSegment *compactend = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isDead());
		checkCUDAError("thrust::partition");
		num_paths = compactend - dev_paths;
		if (0 == num_paths) { iterationComplete = true; }
#elif 2 == COMPACT
		num_paths = StreamCompaction::SharedAndBank::compactNoMalloc_PathSegment(num_paths, dev_paths);
		checkCUDAError("stream compaction");
		//printf("\nnum_paths: %i", num_paths);
		if (0 == num_paths) { iterationComplete = true; }
#else
		printf("\n UKNOWN COMPACT setting \n");
#endif

	}//////////////////////END WHILE


#if 1 == TIMER
	cudaDeviceSynchronize();
	time_point_t time_end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
	float prev_elapsed_time_cpu_milliseconds = static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());
	printElapsedTime(prev_elapsed_time_cpu_milliseconds, "(std::chrono Measured)");
#endif

	// Assemble this iteration and apply it to the image
	num_paths = dev_path_end - dev_paths;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, cam.resolution, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
