#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "stream_compaction\efficient.h"

#define ERRORCHECK 1

// Various rendering feature toggles.
#define ANTIALIAS true					// Toggle antialiasing.
#define CONTIGUOUS false				// Toggle contiguous memory by ray material.
#define CACHE_FIRST true				// Toggle caching the results of the first ray bounce.
#define DOF true						// Toggle depth of field effects.
#define DIRECT_LIGHTING true			// Toggle the direct lighting of the scene.

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

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static int* dev_lights = NULL;
static Material* dev_materials = NULL;
static int* dev_materialIDs_path = NULL;
static int* dev_materialIDs_intersection = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_first = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lights, scene->geoms.size() * sizeof(int));
	cudaMemset(dev_lights, -1, scene->geoms.size() * sizeof(int));

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materialIDs_path, pixelcount * sizeof(int));
	cudaMalloc(&dev_materialIDs_intersection, pixelcount * sizeof(int));

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_first, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
	cudaFree(dev_lights);
  	cudaFree(dev_materials);
	cudaFree(dev_materialIDs_path);
	cudaFree(dev_materialIDs_intersection);
  	cudaFree(dev_intersections);
	cudaFree(dev_first);

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

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// Seed the rng.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		// Implemented toggle-able antialiasing by jittering the ray
		if (CACHE_FIRST) {
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
		} else {

			// Antialiasing is enabled, so add random jitter.
			if (ANTIALIAS) {
				segment.ray.direction = glm::normalize(cam.view
					- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
					- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng)));
			} 
			
			// Depth of field is enabled, so alter the scene based on the given 
			// lens radius and focal distance.
			if (DOF) {

				// Equation per PBRT page 374.
				// Modify ray for depth of field. Wasn't worth copying these to device properly.
				const static float LENS_RADIUS = 1;		// Use this lens radius.
				const static float FOCAL_DISTANCE = 10;	// Use this focal distance.
				if (LENS_RADIUS > 0) {

					// Sample a point on the lens.
					// "Sampling a unit disk" from PBRT 779.
					// Map uniform numbers to [-1, 1].
					glm::vec2 sampledPoint = glm::vec2(0, 0);
					glm::vec2 uOffset = (2.f * u01(rng)) - glm::vec2(1, 1);
					
					// Ignore degeneracy at the origin.
					// Apply concentric mapping to point.
					if (uOffset.x != 0 && uOffset.y != 0) {
						float theta;
						float r;
						if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
							r = uOffset.x;
							theta = (PI / 4.0f) * (uOffset.y / uOffset.x);
						} else {
							r = uOffset.y;
							theta = (PI / 2.0f) - (PI / 4.0f) * (uOffset.x / uOffset.y);
						}
						sampledPoint = r * glm::vec2(std::cos(theta), std::sin(theta));
					}

					// Compute point on plane of focus.
					glm::vec3 focalLength = segment.ray.direction * FOCAL_DISTANCE;
					glm::vec3 focalPoint = segment.ray.origin + focalLength;

					// Update ray effect for lens.
					segment.ray.origin += (cam.right * LENS_RADIUS * sampledPoint.x) 
						+ (cam.up * LENS_RADIUS * sampledPoint.y);
					segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
				}
			}
			
			if (!ANTIALIAS && !DOF) {
				segment.ray.direction = glm::normalize(cam.view
					- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
					- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
			}
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths, 
	PathSegment * pathSegments, Geom * geoms, int geoms_size, 
	ShadeableIntersection * intersections) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++) {
			Geom & geom = geoms[i];

			if (geom.type == CUBE) {
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			} else if (geom.type == SPHERE) {
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		// If the ray misses everything...
		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;

		// The ray hits something otherwise.
		} else {
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
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
	int iter, int num_paths, ShadeableIntersection * shadeableIntersections, 
	PathSegment * pathSegments, Material * materials, int* dev_lights, int num_lights,
	Geom * dev_geoms, int geoms_size) {

	// Get the index and check validity.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0) {

		// Get the intersection of this ray and see what it hit.
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) {

			// Set up the RNG.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			// There was an intersection, retrieve the hit material and color.
			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}

			// Otherwise, bounce around the scene and change color based on the hit material.
			else {
				glm::vec3 t = glm::vec3(0.0f);
				t.t = intersection.t;
				scatterRay(pathSegments[idx], t, intersection.surfaceNormal, material, rng);

				// Direct lighting implementation (by taking a final ray directly to a random 
				// point on an emissive object acting as a light source). Only cast this ray
				// if there exist lights and if this really is the final bounce for a given ray.
				if (num_lights > 0 && pathSegments[idx].remainingBounces == 0) {
					glm::vec3 direction = pathSegments[idx].ray.direction;
					int randomLightIndex = u01(rng) * num_lights;
					int lightGeomIndex = dev_lights[randomLightIndex];
					Geom light = dev_geoms[lightGeomIndex];
					Material lightMaterial = materials[light.materialid];

					// Check for a collision with the randomly-selected light.
					// This piece ripped straight from the other collision-checking.
					float t;
					float t_min = FLT_MAX;
					int hit_geom_index = -1;
					bool outside = true;
					glm::vec3 tmp_intersect;
					glm::vec3 tmp_normal;
					for (int i = 0; i < geoms_size; i++) {
						Geom& geom = dev_geoms[i];
						if (geom.type == CUBE) {
							t = boxIntersectionTest(geom, pathSegments[idx].ray, tmp_intersect, tmp_normal, outside);
						} else if (geom.type == SPHERE) {
							t = sphereIntersectionTest(geom, pathSegments[idx].ray, tmp_intersect, tmp_normal, outside);
						}

						// Compute the minimum t from the intersection tests to determine what
						// scene geometry object was hit first.
						if (t > 0.0f && t_min > t) {
							t_min = t;
							hit_geom_index = i;
						}
					}

					// If the ray hits the light, color it by the light.
					if (hit_geom_index == lightGeomIndex) {
						pathSegments[idx].color *= lightMaterial.color;

					// Otherwise this ray is in shadow to its chosen light.
					} else { 
						pathSegments[idx].color = glm::vec3(0.f);
					}
				}
			}

		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
		} else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < nPaths) {
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
* A predicate for the stream compaction which removes any rays that ran out
* of bounces.
*/
struct notOutOfBounces {
	__host__ __device__
		bool operator() (const PathSegment path) {
		return (path.remainingBounces > 0);
	}
};

/**
* A predicate for the stream compaction which helps count how many lights the
* scene has. The dev_lights array is initialized with sentinels of -1.
*/
struct isALight {
	__host__ __device__
		bool operator() (const int lightCandidate) {
		return (lightCandidate >= 0);
	}
};

// Use a custom kernel to fetch an array of all intersection material IDs.
__global__ void kernExtractMaterialIDs(int numObjects, int* dev_materialIDs, 
	ShadeableIntersection* dev_intersections) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= numObjects) {
		return;
	}

	dev_materialIDs[index] = dev_intersections[index].materialId;
}

// Use a custom kernel to find all geometries that are also lights.
__global__ void kernCountLights(Geom* dev_geoms, int geom_size, Material* dev_materials, 
	int* dev_lights) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= geom_size) {
		return;
	}

	// Check to see if this thread's geometry has a light material.
	Geom lightCandidate = dev_geoms[index];
	Material m = dev_materials[lightCandidate.materialid];
	if (m.emittance > 0.0f) {
		dev_lights[index] = index;
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
    //   * Done below: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Done in shader: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

	// Prepare the first ray out of the camera.
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	// Figure out what the lights are in the scene.
	kernCountLights<<<blocksPerGrid2d, blockSize2d>>>(dev_geoms, hst_scene->geoms.size(),
		dev_materials, dev_lights);

	// Figure out how many lights there are in the scene.
	int* dev_lights_end = thrust::partition(thrust::device, dev_lights,
		dev_lights + hst_scene->geoms.size(), isALight());
	int num_lights = dev_lights_end - dev_lights;
	if (!DIRECT_LIGHTING) {
		num_lights = 0;
	}

	// Timing values.
	int TIMING_RUNS = 1000;
	float tracingTime = 0;
	float shadingTime = 0;
	float contiguousTime = 0;
	float compactionTime = 0;

	// Continue iterating and shooting rays until the scene is built.
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	int num_paths_workable = num_paths;
	bool iterationComplete = false;
	StreamCompaction::Common::PerformanceTimer timer;
	while (!iterationComplete) {

		// Start the timer.
		if (TIMING_RUNS > 0) {
			timer.startGpuTimer();
		}

		// --- PathSegment Tracing Stage ---
		// Shoot ray into scene, bounce between objects, push shading chunks
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// Tracing
		dim3 numblocksPathSegmentTracing = (num_paths_workable + blockSize1d - 1) / blockSize1d;
		
		// If we are set to cache, then record the intersections just computed as 
		// belonging to the camera and reuse them across iterations.
		if (CACHE_FIRST && depth == 0) {

			// If this is the first iteration then the cache must be old so we record the hits.
			// Found out the hard way that iter always comes from main starting as 1, not 0.
			if (iter == 1) {
				computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
					depth, num_paths_workable, dev_paths, dev_geoms,
					hst_scene->geoms.size(), dev_intersections);
				checkCUDAError("trace one bounce");
				cudaMemcpy(dev_first, dev_intersections,
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			
			// Otherwise, the cache is valid so we can go ahead and copy it in as our
			// intersection results.
			} else {
				cudaMemcpy(dev_intersections, dev_first,
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}

		// If we aren't caching or this isn't the first hit, we definitely need to calculate.
		} else {
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
				depth, num_paths_workable, dev_paths, dev_geoms,
				hst_scene->geoms.size(), dev_intersections);
			checkCUDAError("trace one bounce");
		}
		cudaDeviceSynchronize();
		depth++;

		// Record the tracing stage time.
		if (TIMING_RUNS > 0) {
			timer.endGpuTimer();
			tracingTime += timer.getGpuElapsedTimeForPreviousOperation();
			timer.startGpuTimer();
		}

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(iter, 
			num_paths_workable, dev_intersections, dev_paths, dev_materials,
			dev_lights, num_lights, dev_geoms, hst_scene->geoms.size());

		// Record the shading stage time.
		timer.endGpuTimer();
		shadingTime += timer.getGpuElapsedTimeForPreviousOperation();
		timer.startGpuTimer();

		// Compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		if (CONTIGUOUS) {

			// Extract an array of material IDs for every single ray, then use it to sort
			// dev_intersections and dev_paths, kinda like boid index extraction in HW1.
			kernExtractMaterialIDs<<<numblocksPathSegmentTracing, blockSize1d>>>(
				num_paths_workable, dev_materialIDs_path, dev_intersections);
			cudaMemcpy(dev_materialIDs_intersection, dev_materialIDs_path, 
				num_paths_workable * sizeof(int), cudaMemcpyDeviceToDevice);
			thrust::sort_by_key(thrust::device, dev_materialIDs_path,
				dev_materialIDs_path + num_paths_workable, dev_paths);
			thrust::sort_by_key(thrust::device, dev_materialIDs_intersection,
				dev_materialIDs_intersection + num_paths_workable, dev_intersections);

			// Record the time spent making memory contiguous.
			if (TIMING_RUNS > 0) {
				timer.endGpuTimer();
				contiguousTime += timer.getGpuElapsedTimeForPreviousOperation();
				timer.startGpuTimer();
			}
		}

		// Stream compact away all of the terminated paths.
		// You may use either your implementation or `thrust::remove_if` or its cousins.
		// Choosing to use thrust::partition because I want to keep all the paths in dev_paths still.
		// https://thrust.github.io/doc/group__partitioning.html#gac5cdbb402c5473ca92e95bc73ecaf13c
		// I think I need them in there later when the actual image is generated.
		dev_path_end = thrust::partition(thrust::device, dev_paths,
			dev_paths + num_paths_workable, notOutOfBounces());
		num_paths_workable = dev_path_end - dev_paths;
		if (depth >= traceDepth || num_paths_workable <= 0) {
			iterationComplete = true;
		}

		// Record the time spent using compaction to filter terminated rays.
		if (TIMING_RUNS > 0) {
			timer.endGpuTimer();
			compactionTime += timer.getGpuElapsedTimeForPreviousOperation();
		}
	}

	// Print out the current timing averages once enough iterations have passed.
	if (iter > 0 && iter % TIMING_RUNS == 0) {
		printf("Average Times per Iteration by Stage\n");
		printf("Tracing time : %fms\n", (tracingTime / TIMING_RUNS));
		printf("Shading time : %fms\n", (shadingTime / TIMING_RUNS));
		printf("Contiguous time : %fms\n", (contiguousTime / TIMING_RUNS));
		printf("Compaction time : %fms\n", (compactionTime / TIMING_RUNS));
		printf("Total time : %fms\n", ((tracingTime + shadingTime +
			contiguousTime + compactionTime) / TIMING_RUNS));
		tracingTime = 0;
		shadingTime = 0;
		contiguousTime = 0;
		compactionTime = 0;
	}

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
