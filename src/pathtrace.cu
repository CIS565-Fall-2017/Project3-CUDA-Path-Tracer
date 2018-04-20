#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

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
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

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

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

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
__global__ void generateRayFromCamera(Camera cam
    , int iter
    , int traceDepth
    , PathSegment * pathSegments
)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.accumColor = glm::vec3(0.f);
        segment.throughput = segment.color;
        segment.hitSpecularObject = false;

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, index);
        thrust::uniform_real_distribution<float> u_onehalf(-0.5f, 0.5f);

        float xJitter = u_onehalf(rng) * cam.pixelLength.x;
        float yJitter = u_onehalf(rng) * cam.pixelLength.y;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            + glm::vec3(xJitter, yJitter, 0.f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__device__ float SDF(
    glm::vec3 pos
    , enum ImplicitSurfaceType type
    , float scale
)
{
    switch (type)
    {
    case NONE:
        return 0.f; // degenerate case
    case SPHERE_IMPLICIT:
        return min(length(pos) - scale, length(pos - glm::vec3(0.f, 0.5f, 0.f)) - scale * 0.5f);
    case MANDELBULB:
        // This SDF for the Mandelbulb belongs to Inigo Quilez: https://www.shadertoy.com/view/ltfSWn
        const float power = 8.f;
        pos /= scale;

        glm::vec3 w = pos;
        float m = dot(w, w);

        glm::vec4 trap = glm::vec4(abs(w), m);
        float dz = 1.0;

        for (int i = 0; i < 4; i++)
        {
            #if 1
            float m2 = m*m;
            float m4 = m2*m2;
            dz = power*sqrt(m4*m2*m)*dz + 1.0;

            float x = w.x; float x2 = x*x; float x4 = x2*x2;
            float y = w.y; float y2 = y*y; float y4 = y2*y2;
            float z = w.z; float z2 = z*z; float z4 = z2*z2;

            float k3 = x2 + z2;
            float k2 = glm::inversesqrt(k3*k3*k3*k3*k3*k3*k3);
            float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
            float k4 = x2 - y2 + z2;

            w.x = pos.x + 64.0*x*y*z*(x2 - z2)*k4*(x4 - 6.0*x2*z2 + z4)*k1*k2;
            w.y = pos.y + -16.0*y2*k3*k4*k4 + k1*k1;
            w.z = pos.z + -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;
            #else
            dz = 8.0*pow(m, 3.5)*dz + 1.0;

            float r = length(w);
            float b = 8.0*acos(clamp(w.y / r, -1.0, 1.0));
            float a = 8.0*atan(w.x, w.z);
            w = p + pow(r, 8.0) * vec3(sin(b)*sin(a), cos(b), sin(b)*cos(a));
            #endif        

            trap = min(trap, glm::vec4(abs(w), m));

            m = dot(w, w);
            if (m > 4.0)
                break;
        }
        trap.x = m;

        return 0.25*log(m)*sqrt(m) / dz;
    }
    return 0.f;
}

__device__ glm::vec3 ComputeNormal(
    glm::vec3 pos
    , float currDist
    , enum ImplicitSurfaceType type
    , float scale
)
{
    return glm::normalize(glm::vec3(SDF(pos + glm::vec3(1.f * EPSILON, 0.f, 0.f), type, scale) - SDF(pos - glm::vec3(1.f * EPSILON, 0.f, 0.f), type, scale),
        SDF(pos + glm::vec3(0.f, 1.f * EPSILON, 0.f), type, scale) - SDF(pos - glm::vec3(0.f, 1.f * EPSILON, 0.f), type, scale),
        SDF(pos + glm::vec3(0.f, 0.f, 1.f * EPSILON), type, scale) - SDF(pos - glm::vec3(0.f, 0.f, 1.f * EPSILON), type, scale)));
}

__device__ void ComputeIntersectionsHelper(
    PathSegment & pathSegment
    , Geom* geoms
    , int geoms_size
    , ShadeableIntersection & currentIsect
    , int ignoreIndex
)
{
    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        if (i != ignoreIndex)
        {
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == PLANE)
            {
                t = planeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == IMPLICITBOUNDINGVOLUME)
            {
                // Compute intersection w/ the bounding sphere, mark this path segment as currently being in contact with an
                // implicit surface. run a raymarching kernel for all path segments that says
                // if this path segment is the right index && is hitting an implicit surface, while t < tmax, march
                // - also need to compute the max t value for the marching limit
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
            }
        }
    }

    if (hit_geom_index == -1)
    {
        currentIsect.t = -1.0f;
    }
    else
    {
        Geom & hitGeom = geoms[hit_geom_index];

        //The ray hits something
        currentIsect.t = t_min;
        currentIsect.materialId = geoms[hit_geom_index].materialid;
        currentIsect.surfaceNormal = normal;

        if (hitGeom.type == IMPLICITBOUNDINGVOLUME)
        {
            if (outside) // do the raymarch below
            {
                bool newOutside;
                Ray newRay = pathSegment.ray;
                newRay.origin = tmp_intersect + 0.01f * newRay.direction;
                glm::vec3 isectPos, isectNormal;
                float tMax = sphereIntersectionTest(hitGeom, newRay, isectPos, isectNormal, newOutside);
                tMax += t_min;

                // Now raymarch
                const int numSteps = 2000;
                int i = 0;
                float rayT = t_min;
                const float scale = 5.f;
                float distance;
                glm::vec3 currentPos;
                while (rayT <= tMax && i < numSteps)
                {
                    currentPos = pathSegment.ray.origin + rayT * pathSegment.ray.direction;
                    distance = SDF(currentPos - hitGeom.translation, hitGeom.implicitType, scale);

                    if (distance < 0.001f) // very close to the surface
                    {
                        currentIsect.t = rayT;
                        currentIsect.surfaceNormal = ComputeNormal(currentPos - hitGeom.translation
                            , distance
                            , hitGeom.implicitType
                            , scale); // compute implicit normals using gradient method
                        // Material ID is already set for this intersection
                        return;
                    }
                    rayT += distance;
                    i++;
                }

                // If we don't march into anything, intersect with the rest of the scene
                ComputeIntersectionsHelper(pathSegment, geoms, geoms_size, currentIsect, hit_geom_index); // ignore this bounding volume
            }
            else // Already inside, so compute the tMax, march to it, if we don't hit the implicit surface, then compute intersection with the rest of the scene
            {
                bool newOutside;
                Ray newRay = pathSegment.ray;
                newRay.origin = tmp_intersect;
                glm::vec3 isectPos, isectNormal;
                float tMax = currentIsect.t;

                // Now raymarch
                const int numSteps = 2000;
                int i = 0;
                float rayT = 0.0001f;
                const float scale = 5.f;
                float distance;
                glm::vec3 currentPos;
                while (rayT <= tMax && i < numSteps)
                {
                    currentPos = pathSegment.ray.origin + rayT * pathSegment.ray.direction;
                    distance = SDF(currentPos - hitGeom.translation, hitGeom.implicitType, scale);

                    if (distance < 0.001f) // very close to the surface
                    {
                        currentIsect.t = rayT;
                        currentIsect.surfaceNormal = ComputeNormal(currentPos - hitGeom.translation
                            , distance
                            , hitGeom.implicitType
                            , scale); // compute implicit normals using gradient method
                        // Material ID is already set for this intersection
                        return;
                    }
                    rayT += distance;
                    i++;
                }

                // If we don't march into anything, intersect with the rest of the scene
                ComputeIntersectionsHelper(pathSegment, geoms, geoms_size, currentIsect, hit_geom_index); // ignore this bounding volume
            }
        }
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size
    , ShadeableIntersection * intersections
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        ComputeIntersectionsHelper(pathSegments[path_index], geoms, geoms_size, intersections[path_index], -1);
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

__device__ glm::vec3 ComputeDirectLighting(
    int index
    , int iter
    , int nLights
    , int numGeoms
    , PathSegment & pathSegment
    , ShadeableIntersection & shadeableIsect
    , Material * materials
    , Geom * geometry
    , Geom & chosenLight
    , float & pdf_BSDF
    , float & pdf_Light
)
{
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegment.remainingBounces + index * index);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Material & mat = materials[shadeableIsect.materialId];

    // Light Importance Sampling - sample a position on a randomly chosen light in the scene
    int randIndex = min(floor(u01(rng) * (float)nLights), (float)(nLights - 1));
    Geom & lightToSample = geometry[randIndex]; // this is ok b/c we ensure in the scene file that lights are listed first in the list of geometry
    glm::vec3 sampleOnLight = glm::vec3(u01(rng) - 0.5f, u01(rng) - 0.5f, 0); // we assume plane-shaped lights here
    sampleOnLight = multiplyMV(lightToSample.transform, glm::vec4(sampleOnLight, 1.f)); // put the sample into world space

    // Output the chosen light because for MIS, it is needed for BSDF Importance Sampling
    chosenLight = lightToSample;

    /* Compute the various components needed for the LTE */

    glm::vec3 f, isectPos, normal, lightNormal;
    float pdf, absDot;

    // Compute intersection parameters and compute absDot
    isectPos = pathSegment.ray.origin + shadeableIsect.t * pathSegment.ray.direction;
    normal = glm::normalize(shadeableIsect.surfaceNormal);
    glm::vec3 lightSampleDirection = glm::normalize(sampleOnLight - isectPos);
    absDot = abs(glm::dot(lightSampleDirection, normal));

    // Compute the pdf - in steradians (solid angle)!
    Ray newRay;
    newRay.origin = isectPos + normal * 10.f * EPSILON; // the scale determined by trial and error
    newRay.direction = lightSampleDirection;
    lightNormal = glm::normalize(multiplyMV(lightToSample.invTranspose, glm::vec4(0.f, 0.f, 1.f, 0.f)));
    pdf = ComputeLightPDF(lightToSample, lightNormal, newRay);

    // Store light color
    Material & lightMat = materials[chosenLight.materialid];
    glm::vec3 lightColor = lightMat.color * lightMat.emittance;

    // Compute visibility term
    PathSegment psCopy = pathSegment;
    psCopy.ray = newRay;
    ShadeableIntersection siCopy = shadeableIsect;
    ComputeIntersectionsHelper(psCopy, geometry, numGeoms, siCopy, -1);
    float vis = (siCopy.materialId == lightToSample.materialid) ? 1.f : 0.f;

    // Compute f
    switch (mat.bxdf)
    {
    case EMISSIVE:
        pdf_BSDF = 1.0f;
        pdf_Light = 1.0f;
        return pathSegment.color * mat.color * mat.emittance;
    case DIFFUSE:
        pdf_BSDF = glm::dot(normal, lightSampleDirection) / PI;
        f = mat.color / PI;
        break;
    case SPECULAR_BRDF:
        pdf_BSDF = 0.0f;
        f = glm::vec3(0.f); // specular materials are black in direct lighting
        break;
    }

    // This check prevents fireflies from appearing (division by very small value creates blown-out pixels (the scale value determined by trial and error)
    if (pdf <= 5.f * EPSILON)
    {
        pdf_Light = 0.0f;
        return glm::vec3(0.f);
    }
    else
    {
        // Compute LTE
        pdf_Light = pdf;
        return f * lightColor * vis * absDot / pdf;
    }
}

// Computes direct lighting, but performs BSDF Importance sampling rather than light importance sampling
__device__ glm::vec3 ComputeDirectLighting_BSDF(
    int index
    , int iter
    , int nLights
    , int numGeoms
    , PathSegment & pathSegment
    , ShadeableIntersection & shadeableIsect
    , Material * materials
    , Geom * geometry
    , Geom & chosenLight
    , float & pdf_Light
    , float & pdf_BSDF
)
{
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegment.remainingBounces + index * index);

    Material & mat = materials[shadeableIsect.materialId];

    /* Compute the various components needed for the LTE */

    glm::vec3 f, isectPos, normal;
    float pdf, absDot;
    glm::vec3 bsdfSampleRayDir;

    normal = glm::normalize(shadeableIsect.surfaceNormal);

    switch (mat.bxdf)
    {
    case EMISSIVE:
        pdf_BSDF = 1.0f;
        pdf_Light = 1.0f;
        return pathSegment.color * mat.color * mat.emittance;
    case DIFFUSE:
        bsdfSampleRayDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        f = mat.color / PI;
        pdf = glm::dot(normal, bsdfSampleRayDir) / PI;
        break;
    case SPECULAR_BRDF:
        bsdfSampleRayDir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        f = mat.color;
        pdf = 1.0f;
        break;
    }

    isectPos = pathSegment.ray.origin + shadeableIsect.t * pathSegment.ray.direction;
    absDot = abs(glm::dot(bsdfSampleRayDir, normal));

    // Compute the pdf - in steradians (solid angle)!
    Ray newRay;
    newRay.origin = isectPos + normal * 10.f * EPSILON; // the scale determined by trial and error
    newRay.direction = bsdfSampleRayDir;
    pdf_Light = ComputeLightPDF(chosenLight, normal, newRay);

    // Store light color
    Material & lightMat = materials[chosenLight.materialid];
    glm::vec3 lightColor = lightMat.color * lightMat.emittance;

    if (pdf_Light == 0.0f)
    {
        lightColor = glm::vec3(0.f);
    }

    // This check prevents fireflies from appearing (division by very small value creates blown-out pixels (the scale value determined by trial and error)
    if (pdf <= 5.f * EPSILON)
    {
        pdf_BSDF = 0.0f;
        return glm::vec3(0.f);
    }
    else
    {
        // Compute LTE
        pdf_BSDF = pdf;
        return f * lightColor * absDot / pdf;
    }
}

__global__ void DirectLightingIntegrator(
    int nPaths
    , int iter
    , int nLights
    , int numGeoms
    , PathSegment * iterationPaths
    , ShadeableIntersection * shadeableIntersections
    , Material * materials
    , Geom * geometry
)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths)
    {
        iterationPaths[index].remainingBounces = 0; // no more than one bounce in direct lighting!
        if (shadeableIntersections[index].t > 0.0f)
        {
            PathSegment & pathSegment = iterationPaths[index];
            Geom chosenLight; // Needed for MIS, but not in this integrator. Lets us reuse the direct lighting code for both integrators
            float pdfa, pdfb; // Needed for MIS, buyt not in this integrator.
            pathSegment.color = (float)nLights * ComputeDirectLighting(index, iter, nLights, numGeoms, pathSegment, shadeableIntersections[index], materials, geometry, chosenLight, pdfa, pdfb);
        }
        else
        {
            iterationPaths[index].color = glm::vec3(0.f);
        }
    }
}

__global__ void MISIntegrator(
    int nPaths
    , int iter
    , int nLights
    , int numGeoms
    , PathSegment * iterationPaths
    , ShadeableIntersection * shadeableIntersections
    , Material * materials
    , Geom * geometry
)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths)
    {
        PathSegment & pathSegment = iterationPaths[index];
        if (shadeableIntersections[index].t > 0.0f)
        {
            if (pathSegment.remainingBounces == 0)
            {
                return;
            }
            //pathSegment.remainingBounces = 0; // THIS IS FOR DIRECT LIGHTING, WE ARE IN MIS INTEGRATOR ************************************ YOO
            ShadeableIntersection & shadeableIsect = shadeableIntersections[index];

            // Compute global illumination - this is nearly identical to Naive Integrator
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, iterationPaths[index].remainingBounces);
            Material & mat = materials[shadeableIsect.materialId];

            glm::vec3 f, globalIllumColor, isectPos, normal;
            float pdf, absDot;
            Ray sampledRay;

            isectPos = pathSegment.ray.origin + shadeableIsect.t * pathSegment.ray.direction;
            normal = glm::normalize(shadeableIsect.surfaceNormal);

            switch (mat.bxdf)
            {
            case EMISSIVE:
                pathSegment.color = pathSegment.throughput * mat.color * mat.emittance;
                pathSegment.remainingBounces = 0;
                return;
            case DIFFUSE:
                sampledRay.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
                sampledRay.origin = (glm::dot(normal, sampledRay.direction) > 0) ? isectPos + normal * EPSILON : isectPos - normal * EPSILON;
                pdf = glm::dot(sampledRay.direction, normal) / PI;
                absDot = abs(glm::dot(sampledRay.direction, normal));
                f = mat.color / PI;
                pathSegment.remainingBounces--;
                pathSegment.hitSpecularObject = false;
                break;
            case SPECULAR_BRDF:
                sampledRay.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
                sampledRay.origin = (glm::dot(normal, sampledRay.direction) > 0) ? isectPos + normal * EPSILON : isectPos - normal * EPSILON;
                f = glm::vec3(1.f);
                pdf = 1.f;
                absDot = 1.f; // specular objects don't attenuate color by absDot
                pathSegment.remainingBounces--;
                pathSegment.hitSpecularObject = true;
                break;
            }

            // Perform no computations for this iteration
            if (pathSegment.hitSpecularObject)
            {
                pathSegment.ray = sampledRay;
                return;
            }

            globalIllumColor = (pdf <= 10.f * EPSILON) ? glm::vec3(0.f) : f * absDot / pdf;

            // Compute direct lighting
            Geom chosenLight;
            float pdf_BSDF_Light; // BSDF-evaluated pdf using the light-importance sampled direction
            float pdf_Light_BSDF; // Light-evaluated pdf using the BSDF-importance sampled direction
            float pdf_Light; // Light evaluated, light-importance sampled
            float pdf_BSDF; // BSDF evaluated, BSDF-importance sampled
            glm::vec3 directLighting_Light = ComputeDirectLighting(index, iter, nLights, numGeoms, pathSegment,
                shadeableIsect, materials, geometry, chosenLight, pdf_BSDF_Light, pdf_Light);
            glm::vec3 directLighting_BSDF = ComputeDirectLighting_BSDF(index, iter, nLights, numGeoms, pathSegment,
                shadeableIsect, materials, geometry, chosenLight, pdf_Light_BSDF, pdf_BSDF);
            // Compute the proper weights using the various PDFs and power heuristic
            float w_Light = PowerHeuristic(1, pdf_Light, 1, pdf_BSDF_Light);
            float w_BSDF = PowerHeuristic(1, pdf_BSDF, 1, pdf_Light_BSDF);
            glm::vec3 directLightingColor = (w_Light * directLighting_Light + w_BSDF * directLighting_BSDF) * (float)nLights;

            // Set pathsegment properties
            pathSegment.accumColor += directLightingColor * pathSegment.throughput;
            pathSegment.color = pathSegment.accumColor; // so we don't have to have another final gather kernel designed to use accumColor instead of color
            pathSegment.throughput *= globalIllumColor;
            pathSegment.ray = sampledRay;
        }
        else
        {
            pathSegment.color = pathSegment.accumColor;
            pathSegment.remainingBounces = 0;
        }
    }
}

__global__ void NaiveIntegrator(
    int nPaths
    , int iter
    , PathSegment * iterationPaths
    , ShadeableIntersection * shadeableIntersections
    , Material * materials
)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment & pathSegment = iterationPaths[index];
        ShadeableIntersection & shadeableIsect = shadeableIntersections[index];

        if (shadeableIsect.t > 0.0f)
        {
            if (pathSegment.remainingBounces == 0)
            {
                pathSegment.color = glm::vec3(0.f);
                return;
            }

            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, iterationPaths[index].remainingBounces);
            Material & mat = materials[shadeableIsect.materialId];

            /* Compute each part of the LTE */

            glm::vec3 f, finalColor, isectPos, normal;
            float pdf, absDot;

            isectPos = pathSegment.ray.origin + shadeableIsect.t * pathSegment.ray.direction;
            normal = glm::normalize(shadeableIsect.surfaceNormal);

            // Evaluate the BxDF and compute various parts of the LTE
            switch (mat.bxdf)
            {
            case EMISSIVE:
                pathSegment.color *= mat.color * mat.emittance;
                pathSegment.remainingBounces = 0;
                return;
            case DIFFUSE:
                pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
                pathSegment.ray.origin = (glm::dot(normal, pathSegment.ray.direction) > 0) ? isectPos + normal * EPSILON : isectPos - normal * EPSILON;
                pdf = glm::dot(pathSegment.ray.direction, normal) / PI;
                absDot = abs(glm::dot(pathSegment.ray.direction, normal));
                f = mat.color / PI;
                pathSegment.remainingBounces--;
                break;
            case SPECULAR_BRDF:
                pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
                pathSegment.ray.origin = (glm::dot(normal, pathSegment.ray.direction) > 0) ? isectPos + normal * EPSILON : isectPos - normal * EPSILON;
                f = glm::vec3(1.f);
                pdf = 1.f;
                absDot = 1.f; // specular objects don't attenuate color by absDot
                pathSegment.remainingBounces--;
                break;
            }

            finalColor = f * pathSegment.color * absDot / pdf;
            pathSegment.color = finalColor;
            return;
        }
        else
        {
            // For all paths that have no intersection
            pathSegment.color = glm::vec3(0.f);
            pathSegment.remainingBounces = 0;
        }
    }
}

// Needed for Thrust stream compaction
struct is_active
{
    __host__ __device__
        bool operator()(const PathSegment &p)
    {
        return p.remainingBounces > 0;
    }
};

struct compareIntersection
{
    __host__ __device__
        bool operator()(const ShadeableIntersection &s, const ShadeableIntersection &p)
    {
        return s.materialId < p.materialId;
    }
};

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


    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int num_active_paths = num_paths;

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    dim3 numblocksPathSegmentTracing = (num_active_paths + blockSize1d - 1) / blockSize1d;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {

        numblocksPathSegmentTracing = (num_active_paths + blockSize1d - 1) / blockSize1d;
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        computeIntersections << < numblocksPathSegmentTracing, blockSize1d >> > (
            num_active_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        depth++;

        //#define SORT_BY_MATERIAL_TYPE
        #ifdef SORT_BY_MATERIAL_TYPE
        {
            // sort by material type - this slows stuff down unless there are a lot of materials to make the sorting worth it
            thrust::device_ptr<ShadeableIntersection> dev_Thrust_isects(dev_intersections);
            thrust::device_ptr<PathSegment> dev_Thrust_paths(dev_paths);
            thrust::sort_by_key(dev_Thrust_isects, dev_Thrust_isects + num_active_paths, dev_Thrust_paths, compareIntersection());
        }
        #endif

        #define NAIVE
        //#define DIRECT
        //#define MIS

        #ifdef NAIVE
        {
            // Compute naive integration
            NaiveIntegrator << < numblocksPathSegmentTracing, blockSize1d >> > (
                num_active_paths
                , iter
                , dev_paths
                , dev_intersections
                , dev_materials
                );
            checkCUDAError("Naive Integrator");
        }
        #else
        {
            #ifdef DIRECT
            {
                // Compute direct lighting integration
                DirectLightingIntegrator << < numblocksPathSegmentTracing, blockSize1d >> > (
                    num_active_paths
                    , iter
                    , hst_scene->numLights
                    , hst_scene->geoms.size()
                    , dev_paths
                    , dev_intersections
                    , dev_materials
                    , dev_geoms
                    );
                checkCUDAError("Direct Lighting Integrator");
            }
            #else
            {
                #ifdef MIS
                {
                    // Compute MIS integration
                    MISIntegrator << < numblocksPathSegmentTracing, blockSize1d >> > (
                        num_active_paths
                        , iter
                        , hst_scene->numLights
                        , hst_scene->geoms.size()
                        , dev_paths
                        , dev_intersections
                        , dev_materials
                        , dev_geoms
                        );
                    checkCUDAError("MIS Integrator");
        }
                #endif
    }
            #endif
}
        #endif
        cudaDeviceSynchronize();

        #define STREAM_COMPACT // note if you remove this, stuff will break.
        #ifdef STREAM_COMPACT
        {
            // Thrust stream compaction
            PathSegment * compactedPaths = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_active());
            num_active_paths = compactedPaths - dev_paths;
        }
        #endif

        // Update iteration condition
        iterationComplete = num_active_paths == 0 || depth > traceDepth;
    }
    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
