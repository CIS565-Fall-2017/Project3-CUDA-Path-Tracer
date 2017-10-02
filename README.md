CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Name: Meghana Seshadri
* Tested on: Windows 10, i7-4870HQ @ 2.50GHz 16GB, GeForce GT 750M 2048MB (personal computer)


## Project Overview

The goal of this project was to get an introduction to writing a GPU Path Tracer in CUDA.


The following features were implemented (most of which can be toggled with flags in `utilities.h`):

Path Tracing Features:

* Shading kernels with BSDF evaluation for:
	- Ideal diffuse surfaces (using cosine-weighted scatter function)
	- Perfectly specular-reflective surfaces
	- Specular-refractive surfaces
* Naive lighting
* Direct lighting 
* Anti-aliasing
* Depth of field


GPU Features:

* Path continuation/termination using Stream Compaction 
* Sorting rays, path segments, and intersections to be contiguous in memory by materials
* Caching first bounce intersections for re-use across all subsequent iterations 


## Renders

### Naive Lighting 

![](Renders/Naive/cornell.2017-10-01_15-15-21z.5000samp.png)
###### (Run with 5000 samples)


### Direct Lighting 

![](Renders/Direct/transmissiveCube.2017-10-02_01-16-13z.3860samp.png)
###### (Run with 3860 samples)


### Anti Aliasing 


![](Renders/AntiAliasing/transmissiveCube.2017-10-01_21-02-37z.5000samp_AAcheck_noAA.png)
![](Renders/AntiAliasing/transmissiveCube.2017-10-01_21-33-01z.5000samp_AAcheck_AA.png)
###### (Both images run with 5000 samples)

![](Renders/AntiAliasing/cornell.2017-09-28_16-23-12z.67samp_AAcheck_noAA.png)
###### (Run with 67 samples)
![](Renders/AntiAliasing/cornell.2017-09-28_16-24-41z.324samp_AAcheck_AA.png)
###### (Run with 324 samples)


### Depth of field

![](Renders/dof_images/USEtransmissiveCube.2017-10-02_00-54-15z.2518samp_dof_0pt2rad_20focal.png)
###### (Run with 2518 samples, 0.2 radius, 20 focal length)


### Tranmission 

![](Renders/transmissive/USEtransmissiveCube.2017-10-01_23-22-21z.1558samp.png)
###### (Run with 1558 samples)


![](Renders/transmissive/USEtransmissiveCube.2017-10-01_23-35-26z.2364samp_1pt5ior.png)
###### (Run with 2364 samples, 1.5 index of refraction)

![](Renders/transmissive/USEtransmissiveCube.2017-10-01_23-56-06z.2801samp_2pt5ior.png)
###### (Run with 2801 samples, 2.5 index of refraction)

![](Renders/transmissive/USEtransmissiveCube.2017-10-01_23-46-43z.2442samp_5pt5ior.png)
###### (Run with 2442 samples, 5.5 index of refraction)



## Performance Analysis 

All of the following charts and graphs have been tested on one sample with varying ray depths. The scene used is the "TransmissiveCube" scene with the following schematics:

* Naive lighting
* 8 objects (1 transmissive, 1 emissive, 2 reflective, 4 diffuse)


### Stream Compaction 

![](Renders/Charts/sc-chart.PNG)

![](Renders/Charts/sc-graph.PNG)


### Material Sorting

![](Renders/Charts/ms-chart.PNG)

![](Renders/Charts/ms-graph.PNG)


### First Bounce Caching 

![](Renders/Charts/cache-chart.PNG)

![](Renders/Charts/cache-graph.PNG)


## Features to be implemented in the future

* Material sorting by material type (not just ID)
* Multiple importance sampling lighting 
* Microfacet BTDFs
* Texture and bump mapping
* Arbitrary mesh loading and rendering


## Bloopers