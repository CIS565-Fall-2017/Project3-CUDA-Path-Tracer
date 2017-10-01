CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* LINSHEN XIAO
* Tested on: Windows 10, Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 16.0GB, NVIDIA GeForce GTX 970M (Personal computer)

## Features

###Core Features

* Ideal Diffuse surfaces
* Perfectly specular-reflective surfaces
* Stream Compaction
* Material sort
* First bounce intersection cache

###Extra Features

* Refraction with Frensel effects
* Physically-based depth-of-field
* Stochastic Sampled Antialiasing

Refractive index

![](img/cornell_prob.2017-10-01_17-17-39z.5000samp.png)

From left to right(Refractive index): Water(1.33), Olive oil(1.47), Flint glass(1.62), Sapphire(1.77), Diamond(2.42)

(data from https://en.wikipedia.org/wiki/Refractive_index)

Probablity for different materials:

![](img/cornell_prob.2017-10-01_16-28-23z.5000samp.png)

From left to right(Reflective/refractive): 1/0, 0.75/0.25/, 0.5/0.5, 0.25/0.75, 0/1

![](img/cornell_prob.2017-10-01_16-35-29z.5000samp.png)

From left to right(Diffuse/refractive): 1/0, 0.75/0.25/, 0.5/0.5, 0.25/0.75, 0/1

![](img/cornell_prob.2017-10-01_16-42-11z.5000samp.png)

From left to right(Reflective/Diffuse): 1/0, 0.75/0.25/, 0.5/0.5, 0.25/0.75, 0/1

* Physically-based depth-of-field

![](img/cornell3.2017-09-29_19-28-18z.5000samp.png)

|no DOF | With DOF |
|------|------|
|![](img/cornell2.2017-10-01_17-34-12z.5000samp.png) | ![](img/cornell2.2017-10-01_17-40-31z.5000samp.png) |

* Stochastic Sampled Antialiasing