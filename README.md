CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

*  Fengkai Wu
*  Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, Quadro K620 4095MB (Moore 100C Lab)

## Results
![result_img](https://github.com/wufk/Project3-CUDA-Path-Tracer/blob/master/img/final.PNG)

Features:

1. Shading with BSDF evaluation

2. Path termination using stream compaction

3. Toggeable option to cache first bounce and sort path segments by materials

4. Refraction with frenesel effects using Shilick's approximation

5. Stochastic antialiasing

## Analysis

### Antialiasing

Before:
![nojitter_img](https://github.com/wufk/Project3-CUDA-Path-Tracer/blob/master/img/nonjitter.PNG)

After jittering:
![jitter_img](https://github.com/wufk/Project3-CUDA-Path-Tracer/blob/master/img/jitter.PNG)

By adding a uniform random value to the ray, the aliasing effect is removed. As you can see from the picture, the edges of the cube and the wall is smoothened.

### Sorting materials
![sort_img](https://github.com/wufk/Project3-CUDA-Path-Tracer/blob/master/img/sort.PNG)

The sorting is on ray/path arrays with respect to their materials. It is performed right after computing intersections. However it increase the running time primialy due to this addition operation. Making ray/paths contiguous in memory sorting by material does seem to be a good choice. The reason might due to that each path is independent and the kernel does not access each pixel by material type.

### Caching first bounce

![cache_img](https://github.com/wufk/Project3-CUDA-Path-Tracer/blob/master/img/cache.PNG)

The outcome of the first iteration of the pathtracing is cached in device and reused for the subsequent bouncing. The graph above shows that it indeed increase performance but at a constant rate. Reloading the cache for reuse is also a high cost. 

