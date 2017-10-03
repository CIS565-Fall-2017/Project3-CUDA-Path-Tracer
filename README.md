CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xincheng Zhang
* Tested on:
  *Windows 10, i7-4702HQ @ 2.20GHz 8GB, GTX 870M 3072MB (Personal Laptop)


### Description&Features
-------------
* A shading kernel with BSDF evaluation:
* Ideal Diffuse surfaces
* Perfectly specular-reflective surfaces
* Stream Compaction
* Sort by material
* Cache first bounce
* Refraction with Frensel effects 
* Depth of field
* Motion Blur
* Anti-Aliasing


### Result in Progress
-------------
**Fresnel Refraction**
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Fresnel%20Refraction%20comparison.png)
* Left Ball: Ideal diffuse surface. Only diffuse; 
Middle: Reflection =1; 
Right Ball: Refraction = 1;

**Ideal Diffuse surfaces&Perfectly specular-reflective surfaces**
Ideal Diffuse surfaces:
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Cornell%20beginning.png)
Perfectly specular-reflective surfaces:
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/cornell%20specular.png)

**DOF result scene**
Back wall is set to be a mirror; closed box; rendered using 3534 iteration
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/DOF.png)

**DOF comparison**
Original
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Fresnel%20Refraction%20comparison.png)
with DOF
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/cornell%20with%20DOF.png)


**Motion Blur Result Comparison**
The sphere on the left is moving during rendering
original
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Cornell%20without%20motion.png)
with motion blur
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Cornell%20with%20motion.png)

**Anti-Aliasing Result Comparison**
original
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Cornell%20without%20motion.png) 
with Anti-Aliasing
![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Cornell%20with%20AA.png)


### Performance Analysis
-------------
* The data in the following is tested in an open box (without front wall in the scene). There are overall 4 test cases: 1. Nothing; 2. only stream compaction; 3. only sort by material; 4. only cache first bounce. I compare the overall time of 5000 iterations under different conditions to compare their performance.
* The data is tested by modifying the `SORT_MATERIAL`, `CACHE_FIRST_BOUNCE`and `STREAM_COMPACTION` in `pathtrace.cu`.

 STREAM_COMPACTION | SORT_BY_MATERIAL | CACHE_FIRST_BOUNCE | Time for 5000 iterations (s) 
--------------------------|-----------------------|--------------------------|------------------------------
 OFF                      | OFF                   | OFF                      | 208.495    
 ON                       | OFF                   | OFF                      | 586.416    
 OFF                      | ON                    | OFF                      | 1475.642    
 OFF                      | OFF                   | ON                       | 188.064     
**I tried to turn both stream compaction and sort by material on, the result is 957.648. It becomes faster than only using sort by material. I think it's because stream compaction makes the sort faster.

![](https://github.com/XinCastle/Project3-CUDA-Path-Tracer/blob/master/img/Chart.png)


* From the data and chart above, we can tell that in open box test, both stream compaction and sort by material are slower than the naive approach. Only cache first bounce is faster. 
* As for tests using closed box. The stream compaction doesn't really accelerate the program. The overall time is still similar with the data above. Only using stream compaction in a closed box makes the program slower than only using it in an open box. It's because compared with closed box, open box can be better optimized by stream compaction.


