CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Mauricio Mutai
* Tested on: Windows 10, i7-7700HQ @ 2.2280GHz 16GB, GTX 1050Ti 4GB (Personal Computer)

### Overview

#### Introduction

The aim of this project was to implement a simple path tracer that takes advantage of parallelism within a GPU to allow for faster rendering times.

A path tracer is a program that creates an image from a scene made from 3D objects, which can be made to look like they are composed of different materials. The basic path tracing algorithm is as follows:

* Shoot a ray from the camera's viewpoint towards one of the pixels of the image
* For each ray, check if it collides with any of the objects in the scene
* If so, evaluate the light reflected/refracted by that object, and shoot another ray (chosen based on the object's material's probability distribution function) into the scene
* If not, "terminate" this ray and move on to the next one

The path tracer implemented here is very simple, but does showcase some optimizations that can be made to speed it up.

#### Features

Below are this path tracer's main features:

* Support for perfect or imperfect diffuse and specular reflective materials
* Support for loading .OBJ files through tinyOBJ (only vertex positions and normals)

Toggleable features:

* Stream compaction for terminated paths
* Sort paths by material
* Cache first intersection
* Direct lighting for brighter, more convergent images
* Use normals as colors
* Bounding volume culling for .OBJ meshes

#### Optimizations

The following optimizations were implemented:

* Stream compaction for terminated paths
* Sort paths by material
* Cache first intersection
* Bounding volume culling for .OBJ meshes

Below, we take a look at their effect on performance.

### Analysis

#### General comparison in open Cornell box

The following measurements were made in a scene called `cornellMirrors.txt`, which is a basic Cornell box with a reflective sphere and a left wall that has both diffuse and reflective properties. Below is an example render of it, with direct lighting and 5000 samples:

Furthermore, the measurements were taken by rendering images with 5000 samples and a maximum path depth of 8.

![](cornell-base.png);

The graph belows shows the time required to render this scene with different optimizations enabled:

![](graph-base.png)
