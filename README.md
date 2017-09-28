CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* YAOYI BAI (pennkey: byaoyi)
* Tested on: Windows 10 Professional, i7-6700HQ  @2.60GHz 16GB, GTX 980M 8253MB (My own Dell Alienware R3)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

### 1. Part 1 - Core Features
Basic BSDF Feature

### 2. Part 2 - Own Features
#### 1) Direct lighting
#### 2) Realistic Camera Effect

Uncomment

   

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    
    RealisticCamera(pathSegments[index], rng);

At the last of generateRayFromCamera kernel function, and rebuild the project, the result of the effect would be:

 Also, lensRadius  and focalDistance defined at the beginning of pathtrace.cu can be changed according to the translation of objects inside the scene. We can focus on the sphere in the Cornell box:



#### 3) Fresnel Dielectric