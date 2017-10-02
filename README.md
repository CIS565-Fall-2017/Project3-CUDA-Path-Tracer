CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Mauricio Mutai
* Tested on: Windows 10, i7-7700HQ @ 2.2280GHz 16GB, GTX 1050Ti 4GB (Personal Computer)

### Overview

#### Introduction

The aim of this project was to implement a simple path tracer that takes advantage of parallelism within a GPU to allow for faster rendering times.

A path tracer is a program that creates an image from a scene made from 3D objects, which can be made to look like they are composed of different materials. The basic path tracing algorithm is as follows:

* Shoot a ray from the camera's viewpoint towards one of the pixels of the image
* For each ray, check if it collides with any of the objects in the scene
* If so, evaluate the light reflected/refracted by that object, and shoot another ray (chosen based on the object's material's probability distribution function) into the scene
* If not, "terminate" this ray and move on to the next one

The path tracer implemented here is very simple, but does showcase some optimizations that can be made to speed it up.

#### Features

Below are this path tracer's main features:

* Support for perfect or imperfect diffuse and specular reflective materials
* Support for loading .OBJ files through tinyOBJ (only vertex positions and normals)

Toggleable features:

* Stream compaction for terminated paths
* Sort paths by material
* Cache first intersection
* Direct lighting for brighter, more convergent images
* Use normals as colors
* Bounding volume culling for .OBJ meshes

#### Optimizations

The following optimizations were implemented:

* Stream compaction for terminated paths
* Sort paths by material
* Cache first intersection
* Bounding volume culling for .OBJ meshes

Below, we take a look at their effect on performance.

### Analysis

#### General comparison in open Cornell box

The following measurements were made in a scene called `cornellMirrors.txt`, which is a basic Cornell box with a reflective sphere and a left wall that has both diffuse and reflective properties. Furthermore, the measurements were taken by rendering images with 5000 samples and a maximum path depth of 8. Below is an example render of this scene, with direct lighting and 5000 samples:

![](img/cornell-base.png)

The graph belows shows the time required to render this scene with different optimizations enabled:

![](img/graph-base.png)

Here, we can see that only the cache optimization actually reduced the render time -- it was approximately 10 seconds faster than the bare-bones implementation. What could have caused the other two "optimizations" to make the path tracer take so much longer? To help explain this, below is another graph that shows measurements of a single call to the shader kernel, averaged over 100 calls. This means this is measuring only the kernel that determines the color for each intersection, and ignores the cost of other operations, such as sorting lists.

![](img/graph-shader-only.png)

Here, we notice that the bare-bones version without any optimizations and the version that caches the first hit both have very similar runtimes for the shading kernel. This is expected, since caching does not save time on the shading kernel, but when computing the first batch of intersections for each sample.

More importantly, we see that both the stream compaction and sort by material optimizations were effective at reducing the shading kernel runtime. The former optimization reduces the number of threads, and therefore, the number of warps that need to be scheduled and launched, so it is clear why the overall shading kernel runtime is reduced in this case. When sorting by material, we do not reduce the number of threads, but we do group threads such that, in the vast majority of warps, all threads will be processing the same material, greatly reducing warp divergence, which also explains the performance improvement. In my particular implementation, I assign a negative material ID to intersections that do not hit any object, so those intersections are grouped together as well.

Finally, we can see that combining the two optimizations above further reduces the runtime for a single kernel, but not by much more.

From the two graphs above, we can conclude that, although the stream compactions and sort by material optimizations are successful in reducing the runtime of the shader kernel, there is a very large overhead incurred by both optimizations. For the former, it is the stream compaction algorithm itself that makes the runtime of a single path tracing iteration go up significantly. For the latter, the additional `cudaMemcpy` and sorting operations required by my implementation clearly add a heavy load to the runtime. In short, the costs from this overhead far outweighs the improvements in the shading kernel, such that these "optimizations" actually reduce performance. It is likely that, under other circumstances, such as if we had a much larger and complex shader kernel with a greater divergence potential (i.e. conditional branches), then these optimizations would improve runtime, since they could help in mitigating the cost of such a large kernel, while their overhead would be less significant in relative terms.

Interestingly, the caching first intersections optimization, while very simple, was very effective at reducing runtime. This is probably because, instead of targeting our light-weight shading kernel, it instead removes one round of costly intersection checks from each iteration. This meant it had little overhead compared to the other optimizations, but still made for significant gains.

Next, we will look at the performance impact of individual features in more detail.

#### Caching first intersection vs. maximum ray depth

Below is a graph of the time requires to render `cornellMirrors.txt` with 5000 samples at different maximum ray depths. This compares the bare-bones version and the version that caches first intersections.

![](img/graph-cache-depth.pbg)

As we can see, the absolute difference between runtimes is approximately constant, regardless of the maximum ray depth. This makes sense, since we are always saving the same amount of time per iteration -- i.e., the time required to compute the first intersections. 

Another way to look at this data is to compute the relative difference between runtimes, i.e. `1 - (caching runtime) / (bare-bones runtime)`. This gives us the following graph (note Y-axis scale):

![](img/graph-relative-cache-depth.png)

As we can see, the optimization is approximately 20% faster at first, but quickly drops off to about 5% as we reach a more conventional ray depth of 8. Although the 20% speedup at a ray depth of 1 is great, images rendered with this ray depth are devoid of specular effects and only useful if direct lighting is enabled. So, in relative terms, this optimization does not scale well as the ray depth increases.



### Scene file format
