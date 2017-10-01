CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Name: Jiahao Liu
* Tested on: Windows 10, i7-3920XM CPU @ 2.90GHz 3.10 GHz 16GB, GTX 980m SLI 8192MB (personal computer)

Project Description and features implemented
======================

### Project Description

This project is tend to compare running performance difference in computing prefix sum between CPU scan, naive GPU scan, efficient GPU scan and thrust scan.

### Features implemented

![](build/1. Naive with Fresnel.png)

* Naive Path Tracing with Fresnel

![](build/2. Direct Light.png)

* Simple direct light with fake Sample_Li.

![](build/3. Realistic Camera.png)

* Realistic Camera. Focal distance is set on the backwall so the sphere between camera and backwall is blured.

![](build/4. Trace.gif)

* Motion Blur.

Performance Analysis
======================