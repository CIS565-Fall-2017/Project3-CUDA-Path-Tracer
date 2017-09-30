CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yuxin Hu
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### Yuxin Hu
## Depth of Field
![Cornell Box With Depth Of Field 5000 samples](/img/cornell.depthOfField.5000samp.png)
  <p align="center"><b>Cornell Box With Depth Of Field 5000 samples</b></p>
  
## Specular Material with Fresnel Dielectric Material
![Cornell Box With Specular Material 5000 samples](/img/cornell.transmissive.5000samp.png)
  <p align="center"><b>Cornell Box With Specular Material 5000 samples</b></p>
  
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
  
  From the charts above, I can observe that cache first intersections for paths will slightly increase the path tracing speed for all iterations except the first iteration where the cache has not been created yet. Changing the path maximum depth does not seem to affect the performance improvement in my current implementation.
  
  
  
  
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
  

