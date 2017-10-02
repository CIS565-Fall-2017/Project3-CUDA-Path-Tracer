CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yuxin Hu
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### Yuxin Hu
## Code Change
* Project3-CUDA-Path-Tracer/CMakeList.txt: uncomment the stream compaction
* Added stream compaction codes to stream_compaction directory from Project2
* Added testing_helpers.hpp under src/. This file was copied from Project2 for CUDA timer prints
* pathtracer parameter change:
  pathtracer(uchar4* pbo_dptr, int frameNumber, int iteration, bool cacheFirstIntersection, bool enableDepthOfField, float focalPlaneZ, float lenseRadius, bool sortPathByMaterial)
* Added void savePerformanceAnalysis() function in main.cpp to save performance analysis data
* Please look for comments //============= START CUDA Timer==========// and //==================END CUDA Timer===========// to enable to right timer for performance analysis

## Basic Path Tracer
![Cornell Box 5000 samples](/img/cornell.basic.5000samp.png)
  <p align="center"><b>Cornell Box 5000 samples</b></p>
  
![Sphere Light 5000 samples](/img/sphere.basic.5000samp.png)
  <p align="center"><b>Sphere Light 5000 samples</b></p>
  
## Depth of Field
![Cornell Box With Depth Of Field 5000 samples](/img/cornell.depthOfField.5000samp.png)
  <p align="center"><b>Cornell Box With Depth Of Field 5000 samples</b></p>
  
## Specular Material with Fresnel Dielectric Material
![Cornell Box With Specular Material 5000 samples](/img/cornell.transmissive.labled.5000samp.png)
  <p align="center"><b>Cornell Box With Specular Material 5000 samples</b></p>
  
## Anti-aliasing

| No Anti-aliasing             |  With Anti-aliasing |
:-------------------------:|:-------------------------:
![Sphere without anti-aliasing](/img/cornell.sphereWithoutAntiAliasing.2898samp.png)  |  ![Sphere with anti-aliasing](/img/cornell.sphereWithAntiAliasing.2839samp.png)
  
## Performance Analysis
  * Cache the path first intersections
  ![Performance Analysis Cache First Intersections Maximum Depth = 10](/img/performance.cacheFirstIntersection.depth10.PNG)
  <p align="center"><b>Performance Analysis Cache First Intersections Maximum Depth = 10</b></p>
  
  ![Performance Analysis Cache First Intersections Maximum Depth = 8](/img/performance.cacheFirstIntersection.depth8.PNG)
  <p align="center"><b>Performance Analysis Cache First Intersections Maximum Depth = 8</b></p>
  
  ![Performance Analysis Cache First Intersections Maximum Depth = 6](/img/performance.cacheFirstIntersection.depth6.PNG)
  <p align="center"><b>Performance Analysis Cache First Intersections Maximum Depth = 6</b></p>
  
  ![Performance Analysis Cache First Intersections Maximum Depth = 4](/img/performance.cacheFirstIntersection.depth4.PNG)
  <p align="center"><b>Performance Analysis Cache First Intersections Maximum Depth = 4</b></p>
  
  All performance analysis were running on fresnel.txt scene. From the charts above, I can observe that cache first intersections for paths will slightly increase the path tracing speed for all iterations except the first iteration where the cache has not been created yet. Changing the path maximum depth does not seem to affect the performance improvement in my current implementation.
  
  * Sort path segments by material
  ![Performance Analysis Sort Path By Material](/img/performance.sortByMaterial.pathTracing.depth8.PNG)
  <p align="center"><b>Performance Analysis Sort Path By Material</b></p>
  
  ![Performance Analysis Shading Kernal Runtime With/Without Sort By Material](/img/performance.sortByMaterial.shadingKernal.depth1.PNG)
  <p align="center"><b>Performance Analysis Shading Kernal Runtime With/Without Sort By Material</b></p>
  
  ![Performance Analysis Thrust Sort Timing](/img/performance.thrustSort.640000.PNG)
  <p align="center"><b>Performance Analysis Thrust Sort Timing</b></p>
  
  From first plot above I observe that the performance is worse with the sorting by material, which is out of expectation because it is reasonable to think that putting path segments and intersections with same material in contiguous memory will ensure that threads in the same warp all enter the same if condition in scatterRay. So I plotted the second graph, which shows the shading kernal execution time with/without the sort, and it can be shown that although runtime without sort is less stable than runtime with sort, the runtime difference is within 1 millisecond. In the third plot above, it shows the time it takes to run thrust::sort methods over 640000 path segments and intersections, which is over 40 ms, and that is far more than the runtime advantage of shading kernal with sort. I think the advantage of sort will not show until the number of materials in the scene is big enough. Now there are only three different conditions in my shading kernal, the maximum time it takes for a warp with different material to execute will be the sum of three if condition runtime. When the number of materials and number of conditions in shading kernal increase, the maximum time it takes for a warp with different materials will increase, and the advantage of sort will start to show.
  
  * Stream Compaction Analysis
  ![Stream Compaction Remaining Paths Of open scene fresnell.txt](/img/performance.streamCompactionOpenScene.PNG)
  <p align="center"><b>Stream Compaction Remaining Paths Of open scene fresnell.txt</b></p> 
  I can observe that in an open scene, more than 70% of path has been removed from stream compaction after 8 bounces. The reduction of path number is more in the first few bounces comparing to later bounces.
  
  ![Stream Compaction Performance of open scene fresnell.txt](/img/performance.streamCompactionOpenScene.Performance.PNG)
  <p align="center"><b>Stream Compaction Performance Of open scene fresnell.txt</b></p> 
  The total time it takes to complete the kernal runs of one bounce also reduces as less paths are still being traced after stream compaction. Therefore stream compaction does help to increase the performance.
  
  ![Stream Compaction Remaining Paths Comparason between Open Scene and Closed Scene](/img/performance.streamCompactionClosedScene.PNG)
  <p align="center"><b>Stream Compaction Remaining Paths Comparason between Open Scene and Closed Scene</b></p> 
  As we can see, in a closed scene where light cannot escape, every bounce paths will hit some objects, and only those paths that hit light sources will be removed from stream compaction. Comparing to an open scene, there will be less paths trimmed off at each depth.
  
  ![Stream Compaction Performance of Closed Scene](/img/performance.streamCompactionClosedScene.Performance.PNG)
  <p align="center"><b>Stream Compaction Performance of Closed Scene</b></p>
  Similar to open scene, as less paths remaining for further depth, the time it takes to complete a path tracing reduces.
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
## Debugging Process
  * Error 1. Insufficient Space Allocation for Stream Compaction boolean and indice arrays
  ![Insufficient Space Allocation for Stream Compaction](/img/cornell.debug.InsufficientStreamCompactionSpace.50samp.png)
  <p align="center"><b>Insufficient Space Allocation for Stream Compaction</b></p>
  
  I was using the stream compaction I wrote in last project to remove terminated rays from ray pool. However the stream compaction I wrote last time assumes the boolean and indices arrays allocated on GPU has a size that is power of 2. For this project, the default image is 800*800, which initializes a pool rays with 640000 threads, and it is not a power of 2 number. When I initialized the and allocated space for boolean and indices array on GPU for stream compaction, I allocated 640000 * size(dataType) for both arrays, which will have a an error during the first round of stream compaction in efficient scan process. This will result the rays in the upper part of image being trimmed off for subsequent tracing.
  
  * Error 2. Did not offset the origin for new ray.
  ![No Offset on New Ray Origin](/img/cornell.debug.pureDiffusive.png)
  <p align="center"><b>No Offset on New Ray Origin</b></p>
  
  I did not give an offset for the new ray origin along the new ray direction, then I produced above image.
  
  * Error 3. Normalize the incoming light direction and surface normal wrongly when calculating the cosTheta between two direction.
  ![Normalize the incoming light direction and surface normal wrongly](/img/cornell.debug.transmissive.131samp.png)
  <p align="center"><b>Insufficient Space Allocation for Stream Compaction</b></p>
  In tranmissive material bsdf calculation, I needed to use Snell's Law to determine the refraction direction. The steps I took are taken referenced from University of Pennsylvania Spring 2017 CIS561 Advanced Computer Graphics Projects:
  
  Step1: Calculate the cosine angle between incoming light and surface normal
  float cosThetaI = glm::clamp(glm::dot(glm::normalize(incomingLightDir), glm::normalize(normal)), -1.0f, 1.0f);
  
  Step2: Calculate the sine angle between incoming light direction and surface normal
  float sinThetaI = std::sqrt(max(0.0f, 1 - cosThetaI*cosThetaI));
  
  Step3: Using Snell's Law to calculate the sine angle between refraction direction and surface normal.
  float sinThetaT = etaIn / etaOut*sinThetaI;
  
  In my implementation I mis-typed the equation in step1 as: 
  float cosThetaI = glm::clamp(glm::normalize(glm::dot(incomingLightDir, normal)), -1.0f, 1.0f);
  Which always results cosThetaI in 1. Then in step2, the sinThetaI will be 0. In step3, I will get an undefined number by deviding etaIn with 0, and there is a black ring in the transmissive ball.
  

