CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

Ju Yang 

### Tested on: Windows 7, i7-4710MQ @ 2.50GHz 8GB, GTX 870M 6870MB (Hasee Notebook K770E-i7)
![result](pic/aaa.gif)

## TODOs finished: 
  ### 1. naive BRDF of reflection

I modified the scatterRay function in interaction.h to implement the basic BRDF

![cornell](pic/cornell.2017-10-02_07-42-05z.1009samp.png)

This picture is rendered with about 1000 iterations. 

Reflection parameter in scene files now has a meaning. 

If REFL is 0, that means it does not color any light. 

If REFL is between 0 and 1, it will color the light that hits it. The greater REFL is, the stronger effect it has.

I also modified the scene file "cornell.txt", adding numbers to the REFL from material 1 to material 3.

  ### 2. naive Refraction

I added a branch in scatterray function to implement the refraction. 

![cornell](pic/cornell.2017-10-02_07-17-23z.2422samp.png)

This picture is rendered with about 1000 iterations, in cornell2.txt, with an additional glass ball. 

Refraction parameter in scene file now determines how "refractive" the material is. 

In basic path-tracing, there's only 1 ray we could have for each pixel in each iteration. So when hitting the surface, we must decide whether it should refract or reflect. 

float a = u01(rng)*m.hasReflective / m.hasRefractive; //decide whether this time it will bounce or travel through the glass

If a is larger than 0.5, then the light should go reflective. Or otherwise, it should go refractive. 

Besides, about Snell Law, we need to know whether the light is coming into/out the glass. 

So I added bool isinglass; in the PathSegment struct in sceneStructs.h 

In generateRayFromCamera, we set it to false. 

If it is false, and the ray hits a glass, then we know it is travelling from air into glass. And we will turn it to true.

If it is true, and the ray hits a glass, then we know it is travelling from glass to air. And we will turn it to false.

Besides, when travelling out of the glass, the snell parameter should be (1/snell).
