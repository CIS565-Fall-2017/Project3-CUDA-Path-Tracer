CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Jonathan Lee
* Tested on: Tested on: Windows 7, i7-7700 @ 4.2GHz 16GB, GTX 1070 (Personal Machine)

# Overview

In this project, I was able to implement a basic Path Tracer in CUDA.

100,000 samples, 8 bounces (~1h30m)
![](img/materials100000samp.png)

Features:
- The Cornell Box
- [Depth of Field](#depth-of-field)
- [Anti-Aliasing](#anti-aliasing)
- [Materials](#materials)
    - Diffuse*
    - Specular*
    - Refractive
    - Subsurface 
- [Shading Techniques](#shading-techniques)
    - Naive*
    - Direct Lighting
- Stream Compaction for Ray Termination*
- Material Sorting*
- First Bounce Caching*

[Analysis](#analysis)

(*) denotes required features.

# Results and Renders

## Depth of Field

No Depth of Field             |  Depth of Field
:-------------------------:|:-------------------------:
![](img/nodof_cornell.2017-09-29_04-40-15z.5000samp.png) Lens Radius: 0 & Focal Distance: 5  |  ![](img/dof_cornell.2017-09-29_04-28-53z.5000samp.png) Lens Radius: .2 & Focal Distance: 5 |

## Anti&#8208;Aliasing

NO AA      |  AA
:-------------------------:|:-------------------------:
![](img/noaa.png)   |  ![](img/aa.png)  |

You can definitely see the jagged edges are more profound on the left since AA is turned off. AA was achieved by randomly offsetting/jittering the pixel at each iteration.

## Materials

![](img/materials10000desc.png)

### Specular
Perfectly Specular   |  Pink Metal
:-------------------------:|:-------------------------:
![](img/cornell.2017-10-01_07-20-52z.5000samp.png)      | ![](img/cornell.2017-10-01_07-40-20z.5000samp.png)  |

### Transmissive
Refractive Index: 1.55    |  Refractive Index: 5
:-------------------------:|:-------------------------:
![](img/1.55iorcornell.2017-10-01_07-24-56z.5000samp.png)      | ![](img/ior5cornell.2017-10-01_07-17-22z.5000samp.png)  |



|   |Subsurface Scattering |  |
| ------------- |:-------------:| -----:|
| ![](img/subsurface/backleft.png)      | ![](img/subsurface/backright.png) | ![](img/subsurface/frontright.png) |

Density: .9

At each ray intersection with the object, the ray gets scattered through the object causing a random direction for the next bounce. A direction is chosen by spherical sampling. A distance is also sampled using `-log(rand(0, 1)) / density`.

![](img/cornell.2017-10-01_15-40-44z.50000samp.png)

Here's the same scene but with diffuse and transmissive materials.

Diffuse             |  Transmissive
:-------------------------:|:-------------------------:
![](img/cornell.2017-10-01_16-27-12z.5000samp.png)  |  ![](img/cornell.2017-10-01_16-35-00z.5000samp.png)

## Shading Techniques

Naive             |  Direct Lighting
:-------------------------:|:-------------------------:
![](img/naive_cornell.2017-09-23_19-25-09z.1285samp.png)  |  ![](img/direct_cornell.2017-09-29_06-05-30z.5000samp.png)

When a ray hits an object in the scene, we sample its BSDF and shoot a second ray towards the light. We sample a random position on the light and determine its contribution. If there is another object in between the original object and the light, then the point is in shadow, otherwise we account for the light's contribution. Direct lighting only does a single bounce as opposed to the naive integrator.

*This is based on my CIS561 implementation of the direct lighting integrator.

# Analysis

# Future Work

- Full Lighting/MIS
- Photon Mapping
- Acceleration Structures
- OBJ Loader

# References
- [CIS 561 Path Tracer](https://github.com/AgentLee/PathTracer)
- [PBRT](https://github.com/mmp/pbrt-v3)
- Subsurface Scattering
    - [http://www.davepagurek.com/blog/volumes-subsurface-scattering/]()
    - [https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering]()
- [https://www.scratchapixel.com/]()

# Bloopers

### Naive 
Unless noted otherwise, I don't entirely remember how most of these happened other than messing around in `scatterRay` and depth values.

![](img/bloopers/cornell.2017-09-23_14-30-34z.59samp.png)

![](img/bloopers/cornell.2017-09-23_15-31-20z.166samp.png)

![](img/bloopers/cornell.2017-09-22_21-18-46z.223samp.png)

I assumed the ray was coming from the camera to the object so I negated the outgoing ray which would definitely affect the reflection.

![](img/bloopers/cornell.2017-09-22_05-16-02z.436samp.png)

### Sorting fails

First attempt at sorting the paths by material. I did things on the CPU and also didn't update the paths.

![](img/bloopers/cornell.2017-09-23_22-06-11z.11samp.png)

Forgot to sort the materials before shading. :unamused:

![](img/bloopers/cornell.2017-09-24_03-01-57z.51samp.png)

### Direct Lighting

I think for this one I added an extra lighting term.
![](img/bloopers/cornell.2017-09-27_21-24-41z.296samp.png)

I inverted the shadow ray, so everything was in shadow.
![](img/bloopers/cornell.2017-09-27_19-19-43z.229samp.png)