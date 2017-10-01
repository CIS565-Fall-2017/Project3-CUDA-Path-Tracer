CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Charles Wang
* Tested on: Windows 10, i7-6700K @ 4.00GHz 16GB, GTX 1060 6GB (Personal Computer)

![](img/loris.gif)

**Project Overview and Goals**

The goal of this project is to implement a simple GPU Path Tracer.

The features included in this project are:
 - Naive Path Tracing ("BRDF" sampled lambert and perfect specular)
 - BVH (Bounding Volume Hierarchy) Acceleration 
 - OBJ loading
 - Stream compaction for terminating rays and free-ing up threads
 - Depth of Field
 - Naive Anti-Aliasing

Performance Analysis and more implementational details to come!