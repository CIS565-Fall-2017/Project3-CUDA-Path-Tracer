# **University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3:**

# **CUDA Path Real Time Path Tracer**





Tested on: Windows 10, Intel Core i7-7700HQ CPU @ 2.80 GHz, 8GB RAM, NVidia GeForce GTX 1050

 ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/github/issues-raw/badges/shields/website.svg) ![CUDA 8.0](https://img.shields.io/badge/CUDA-8.0-green.svg?style=flat)  ![Platform](https://img.shields.io/badge/platform-Desktop-bcbcbc.svg)  ![Developer](https://img.shields.io/badge/Developer-Youssef%20Victor-0f97ff.svg?style=flat)




- [Features](#features)



- [In-Depth](#indepth)



- [Blooper](#blooper)
 

 

____________________________________________________


 
The goal of this project was to run an algorithm that clears out all zeros from an array on the GPU using CUDA. This parallel reduction is done using the scan algorithm that computes the exclusive prefix sum. I also implemented a parallelized Radix Sort using the exclusive prefix sum algorithm developed.



### Things Done

#### Core Features

 - [x] Shading for Ideal Diffuse Surfaces
 - [x] Shading for Perfectly Specular Surfaces
 - [x] Early Ray Termination Using Stream Compaction
 - [x] Caching of First Bounces

 #### Spicy Features
  - [x] Shading for Transmissive Surfaces (maybe)
  - [x] Work-efficient Stream Compaction Using Shared Memory Across Multiple Blocks
  - [x] Environment Maps
  - [x] Direct Lighting
  - [x] Multiple Importance Sampling
  - [x] Stochastic Antialiasing
  - [x] Arbitrary Mesh Loading
  - [ ] Kd-Tree & Stackless Kd-Tree Traversal on the GPU
  - [ ] [Physically Accurate Lens Flares](https://placeholderart.wordpress.com/2015/01/19/implementation-notes-physically-based-lens-flares/)
  - [ ] [Specular BRDF with Microstructure BRDF](https://people.eecs.berkeley.edu/~lingqi/publications/paper_glints2.pdf)
  - [ ] Texture Mapping (maybe)
  - [ ] Depth of Field/Cooler Lens Effects such as bloom
 
![Cornell Box](/img/cornellOBJ.png)

Multiple Importance Sampling, 7500 Samples, 1080x1080 px, OBJ dodecahedron mesh.



### In-Depth:

#### Transmissive Surfaces

This uses basic `glm::refract` so it should work. For some reason it doesn't. I should change the implementation of it, but for now, maybe I can get half-credit points. To try out a scene with it simply set a material's bsdf to 2.
`TODO: Show picture`

#### Work Efficient Scan Using Shared Memory Across Multiple Blocks

This took me like 5 days to do, I ended up simply using thrust's partition, but well, that's programming for you. The benefit to my implementation is that it works all the way up to `2^24` as opposed to some others' implementation which fails before that.
`TODO: Show performance comparison`

#### Environment Maps:

![inverted](/img/envMap.png)
An environment map with a purely specular sphere in the middle.

![inverted](/img/envMapDiff.png)
An environment map with a purely diffuse sphere in the middle.

In the sample images above, you can clearly see the effect the environment map has with specular surfaces. With the purely specular surface it sort of looks like there is some bloom effeect showing as well. I think that is really cool.

The second image with a diffuse surface is supposed to be a 98% white sphere. But as you see, as in the real world, the environment also reflects light onto the object. The result is the beautifully shaded sphere you see.

In the following images you will see how the environment maps affect the lighting of a cornell box scene with a specular sphere in the middle:

![cornellBoxNoEnvMap](/img/cornellSpec.png)
Normal cornell box scene with no environment map.

![cornellBoxNoEnvMap](/img/cornellEnvSpec.png)
Normal cornell box scene with environment map added.

The environment map adds better lighting to the scene.


#### Direct Lighting

This took a while to get completely right, it uses a light-based sample to shade the entire scene. As such, there are no reflections on objects. The result however is a much more nicely converged scene.

`TODO: Show Pic`

#### Multiple Importance Sampling

When you combine the light based sampling and the bsdf-based sampling, you get what you saw in the representative image at the start: a completely converged. Here are some more pictures:

![mis-sample1](/img/cornellTwo.2017-10-01_21-11-00z.5000samp.png)

`TODO: Add More Images`

#### Stochastic Anti-Aliasing:

This Feature was fairly simple to implement yet took a lot to perfect. This feature does not work with first-bounce caching unless you jitter the ray direction after you generate the ray, which is possible and might be a `TODO`, but given the lack of time, it will probably remain an idea. Here is what happens when I try anti-aliasing with caching turned on:

![bad-aliasing](/img/cornell.2017-10-01_18-30-30z.5000samp.png)

With my cached first bounce, there are jagged edges surrounding everything as all the rays have been jittered, but they do not change across iterations, so the random jitterness stays and remains prominent. As such the picture looks very weird. Here's what it looks like without anti-aliasing at all:

![no-aliasing](/img/cornellTwo.2017-10-01_09-02-39z.5000samp.png)

The image here has very rough aliased edges. Here is what it looks like with everything fixed:

![aliasing](/img/cornellTwo.2017-10-01_21-11-00z.5000samp.png)

Now the image is much smoother.

#### Arbitrary Mesh Loading

Took me forever to get this to work, triangle intersection and intersection testing on the GPU caused a lot of problems. Eventually they were all resolved, but still. Here is the beautiful representative image once again with a stretched out cyan dodecahedron model:

![Cornell Box](/img/cornellOBJ.png)


### Bloopers / Lessons Learned

This is a blooper I got while trying to get MIS to work
![inverted](/img/bloopers/inverted.png)

Another blooper I got where I was sampling the light weirdly

![the-v](/img/bloopers/the-v.png)

My favorite Blooper, which also shows Direct Ligting Rays being sampled:

![starry](/img/bloopers/the-v.png)![inverted](/img/bloopers/starry.gif)

