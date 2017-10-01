CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* YAOYI BAI (pennkey: byaoyi)
* Tested on: Windows 10 Professional, i7-6700HQ  @2.60GHz 16GB, GTX 980M 8253MB (My own Dell Alienware R3)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

## 1. Part 1 - Core Features
####**Basic BSDF Feature.** 

 - **BSDF**:  a simple Lambert, specular and also mixture of Lambert and specular material in **ScatterRay**;
 - **Steam Compaction**: through thrust::remove to remove the 0 remaining bounces inside pathSegments;
 - **Material Sorting**: through thrust::sort_by_key to compact pathSegments with the same material together.

The Cornell box are Lambert material, and the sphere at the center is a mixture of both Lambert and perfect specular material, therefore it has some reflectivity and also roughness. 

**Cornell.txt** after 5000 iterations:
![enter image description here](https://lh3.googleusercontent.com/-i2Wx3KM3VGc/Wc8ElRgYFCI/AAAAAAAAA6s/RT3osXf1LLc67gfhvPe51cIgmMRUZL38gCLcBGAs/s0/cornell.2017-09-30_02-37-01z.5000samp.png "cornell.2017-09-30_02-37-01z.5000samp.png")

**Mirror.txt** a specular and Lambert mixture sphere at the center, after 500 iterations:
![enter image description here](https://lh3.googleusercontent.com/-zhM-jUWEfnE/Wc8GCnjUv-I/AAAAAAAAA68/FGqjPM4225c88tWV9qbfbIynRMweThMWgCLcBGAs/s0/cornell.2017-09-30_02-42-46z.5000samp.png "cornell.2017-09-30_02-42-46z.5000samp.png")

## 2. Part 2 - Own Features
### 1) Direct lighting
**cornell.txt** after 5000 iterations:
![enter image description here](https://lh3.googleusercontent.com/-HP8DYyMJ-lo/WdDvhCkQvxI/AAAAAAAAA8o/VlEKIv-vmycPnpmsbdQjqiBYjPfBeNPcQCLcBGAs/s0/cornell.2017-10-01_03-59-22z.5000samp.png "cornell.2017-10-01_03-59-22z.5000samp.png")


### 2) Realistic Camera Effect

Uncomment

   

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    
    RealisticCamera(pathSegments[index], rng);

At the last of **generateRayFromCamera** kernel function, and rebuild the project, the result of the effect would be:

 Also, **lensRadius**  and **focalDistance** defined at the beginning of pathtrace.cu can be changed according to the translation of objects inside the scene. We can focus on the sphere in the Cornell box:

**Cornell.txt** after 5000 iterations:
![enter image description here](https://lh3.googleusercontent.com/-HJFEbFjS1x8/Wc-8C9pIZDI/AAAAAAAAA7g/XXqN69WTs80SHLmExtIDW3BH23WevDg5wCLcBGAs/s0/cornell.2017-09-30_15-37-14z.5000samp.png "cornell.2017-09-30_15-37-14z.5000samp.png")

**Realistic.txt** after 5000 iterations:
![enter image description here](https://lh3.googleusercontent.com/-JbpKdBJO3so/Wc-9ZIgJDtI/AAAAAAAAA7w/EBlwHkN8TrYKAkc-Q_-oL5GAz6Pxx-s9QCLcBGAs/s0/cornell.2017-09-30_15-44-36z.5000samp.png "cornell.2017-09-30_15-44-36z.5000samp.png")

**Realistic1.txt** after 500 iterations:
![enter image description here](https://lh3.googleusercontent.com/-Rzb9wMWyYMs/Wc-_Pv7f-4I/AAAAAAAAA8A/dIh6GjxvWIY8XFKpYBBkyej259_pTCyOQCLcBGAs/s0/cornell.2017-09-30_15-53-57z.5000samp.png "cornell.2017-09-30_15-53-57z.5000samp.png")

### 3) Fresnel Dielectric
To test Fresnel effect, please use the **glass.txt** in scene files. 
This scene is basically the same as the Cornell box scene, however, the sphere at the center is a glass ball, which has reflection and refraction. The Fresnel implementation here is quite naive, it simply add up Fresnel specular 



## 3. Performance Analysis 
