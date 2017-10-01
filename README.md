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
 - Material Sorting (to avoid divergence)

Performance Analysis and more implementational details to come!

**Naive Path Tracing**

**BVH Acceleration**

![](img/bvh_table.png)
![](img/bvh_graph_construct.png)
![](img/bvh_graph_iteration.png)

![](img/bvh_example.png)\
1000 samples per pixel, ~20,000 triangles, BVH Depth 14

**Stream Compaction for Terminating Rays**

![](img/compact_table.png)
![](img/compact_graph.png)

**Anti-Aliasing**

![](img/aa_1.png)
![](img/aa_2.png)

**Material Sorting**

![](img/matsort_table.png)