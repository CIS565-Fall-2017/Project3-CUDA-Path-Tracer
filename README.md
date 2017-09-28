CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Mohamad Moneimne
* Tested on: Windows 10, Intel Xeon @ 3.70GHz 32GB, GTX 1070 8120MB (SigLab Computer)

### Overview

In this project, I was able to implement a simple CUDA Monte Carlo path tracer. The overall goal of the project was to understand how a typically CPU-based application, such as path tracing, might be implemented on the GPU. Of course, path tracing is a highly parallel algorithm if you think about it. For every pixel on the image, we cast a ray into the scene and track its movements until it reaches a light source, yielding a color on our screen. In this project, every pixel's ray is dispatched as its own thread, for which intersections and material contributions are processed on the GPU. 

![](img/cornell.gif)

### A Naive Path Tracer

To begin with, I simply stuck with rendering basic shapes without any additional features. These included rendering diffuse and specular materials with light contributions from area lights. As expected of a Monte Carlo path tracer, samples are processed in sequence. For each sample, a kernel casts a ray for each pixel into the scene and bounces up to some terminal depth. If an ray interescts a light source or nothing, the ray will either return an accumulated color along bounces or black, respectively. If a ray bounces around the scene and fails to reach a light source by the maximum depth condition, the sample's contribution is void and black is returned. With this, we can create simple images such as this Cornell Box with 5,000 samples.

![](img/cornell_naive_5000.png)

### Stream Compaction for Ray Culling

An important observation can be made about our naive path tracer. At each depth of a single sample iteration, we always dispatch a kernel with a number of threads that equals the number of pixels being processed. This is wasteful! Consider a ray that misses the scene completely when it is first cast into the scene. We know this ray should return black, but the ray is still dispatched in a kernel at every depth (up until a maximum depth). In order to remedy this, we can use stream compaction at every bounce to cull rays that have already determined their color contribution. This means we will have fewer threads to dispatch for every depth of a single sample, which will allow us to keep more warps available and finish faster at increasing depths.

We expect this to give us a drastic improvement in speed, but here are the timings of a single sample with and without stream compaction on the Cornell Box scene.

![](img/table1.png)

![](img/figure1.png)

As can be seen here, stream compaction ends up being slower! This is likely because there is a significant overhead to doing stream compaction compared to not. One observation that we can make is that with stream compaction, the simulation becomes much faster with increasing depth compared to without stream compaction. We only allow our rays to bounce up to 12 times, but in move productions, rays are likely allowed to bounce much more in order to account for global illumination as best as possible. With a larger depth cap, we would likely see the stream compaction at large depth values have faster speeds than without.

### Material Sorting

