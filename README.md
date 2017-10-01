CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Aman Sachan
* Tested on: Windows 10, i7-7700HQ @ 2.8GHz 32GB, GTX 1070(laptop GPU) 8074MB (Personal Machine: Customized MSI GT62VR 7RE)

## Overview

In this project, I implemented a simple CUDA Monte Carlo path tracer. Although path tracers are typically implemented on the CPU, path tracing itself is a highly parallel algorithm if you think about it. For every pixel on the image, we cast a ray into the scene and track its movements until it reaches a light source, yielding a color on our screen. Each ray is dispatched as its own thread, for which intersections and material contributions are processed on the GPU.

_More Feature filled Path Tracer I made but on the CPU can be found [here](https://github.com/Aman-Sachan-asach/Monte-Carlo-Path-Tracer)._

![](img/StreamCompactionDepthTest.png)

### Introduction to Monte Carlo Path Tracing

Monte Carlo Path tracing is a rendering technique that aims to represent global illumination as accurately as possible with respect to reality. The algorithm does this by approximating the integral over all the illuminance arriving at a point in the scene. A portion of this illuminance will go towards the viewpoint camera, and this portion of illuminance is determined by a surface reflectance function (BRDF). This integration procedure is repeated for every pixel in the output image. When combined with physically accurate models of surfaces, accurate models light sources, and optically-correct cameras, path tracing can produce still images that are indistinguishable from photographs.

Path tracing naturally simulates many effects that have to be added as special cases for other methods (such as ray tracing or scanline rendering); these effects include soft shadows, depth of field, motion blur, caustics, ambient occlusion, and indirect lighting, etc.

Due to its accuracy and unbiased nature, path tracing is used to generate reference images when testing the quality of other rendering algorithms. In order to get high quality images from path tracing, a large number of rays must be traced to avoid visible noisy artifacts.

## Features

![](img/GPUPathTracer_AllFeaturesTiming.png)
![](img/GPUPathTracer_featureComparison.png)

### Naive Integration Scheme

A Naive brute force approach to path tracing. This integrator takes a long time and a large number of samples per pixel to converge upon an image. There are things that can be done to improve speed at which the image converges such as Multiple Important Sampling but these add biases to the image. Usually these biases are very good approximations of how the brute force approach would result. I talk more about Multiple Important Sampling [here](https://github.com/Aman-Sachan-asach/Monte-Carlo-Path-Tracer).

### Stream Compaction

For each iteration of our path tracer we dispatch kernels that do some work. At each depth of an iteration, we always dispatch a kernel with a number of threads that equals the number of pixels being processed. This is wasteful! After every ray bounce there are a number of rays that terminate. This can be because they exit the scene without hitting anything in it, or they can hit a light and terminate. We no longer need to process those rays and issue threads for them the but the ray is still dispatched in a kernel at every depth (up until a maximum depth). In order to remedy this, we can use stream compaction at every bounce to cull rays that we know have terminated and are no longer active. This means we will have fewer threads to dispatch for every depth of a single sample, which will allow us to keep more warps available and finish faster at increasing depths.

One would expect this to give us a drastic improvement in speed, but here are the timings of a single sample with and without stream compaction on the Cornell Box scene.

![](img/StreamCompactionDepthTest.png)
![](img/GPUPathTracer_StreamCompaction.png)

As can be seen here, stream compaction ends up being slower! This is likely because there is a significant overhead to doing stream compaction compared to not. One observation that we can make is that with stream compaction, the simulation becomes much faster with increasing depth compared to without stream compaction. We only allow our rays to bounce up to 12 times, but in movie productions, rays are likely allowed to bounce much more in order to account for global illumination as best as possible. With a larger depth cap, we would likely see the stream compaction at large depth values have faster speeds than without.

### Material Sorting



### First Bounce Caching



### Anti-Aliasing



### Depth of Field



### Direct Lighting Integration Scheme


