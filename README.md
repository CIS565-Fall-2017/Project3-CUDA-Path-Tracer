

CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture**

* Mariano Merchante
* Tested on
  * Microsoft Windows 10 Pro
  * Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 2601 Mhz, 4 Core(s), 8 Logical Processor(s)
  * 32.0 GB RAM
  * NVIDIA GeForce GTX 1070 (mobile version)

## Details
This project implements a physically based pathtracer through the use of CUDA and GPU hardware. It has basic material and scene handling and supports meshes while keeping an interactive framerate. The main features are:

* Brute force path tracing
* Diffuse, perfectly specular reflective and transmissive materials
  * An approximated translucence brdf for thin walled materials such as paper
* Filmic tonemapping with user editable vignetting
* Depth of Field with arbitrary aperture shape
* Texture mapping and one procedural texture
* Normal mapping
* Mesh loading with an accelerated kd-tree structure that runs in GPU
* Antialiasing
* Gaussian filtering of path samples


## Feature description and analysis

### Core path tracer
The brute force algorithm uses a sequence of compute kernels that intersect and accumulate samples from the scene. After each iteration, it compacts and discards paths that 
