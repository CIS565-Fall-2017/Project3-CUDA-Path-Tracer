CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Jonathan Lee
* Tested on: Tested on: Windows 7, i7-7700 @ 4.2GHz 16GB, GTX 1070 (Personal Machine)

# Overview

In this project, I was able to implement a basic Monte Carlo Path Tracer in CUDA.

Features:
- The Cornell Box
- Depth of Field
- Anti-Aliasing
- Materials
    - Diffuse*
    - Specular*
    - Refractive
    - Subsurface 
- Integration/Shading Techniques
    - Naive*
    - Direct Lighting
- Stream Compaction for Ray Termination*
- Material Sorting*
- First Bounce Caching*

(*) denotes required features.

# Results and Renders

## Depth of Field

No Depth of Field             |  Depth of Field
:-------------------------:|:-------------------------:
![](img/nodof_cornell.2017-09-29_04-40-15z.5000samp.png) Lens Radius: 0 & Focal Distance: 5  |  ![](img/dof_cornell.2017-09-29_04-28-53z.5000samp.png) Lens Radius: .2 & Focal Distance: 5 |

## Anti-Aliasing

NO AA      |  AA
:-------------------------:|:-------------------------:
![](img/noaa_cornell.2017-09-29_05-52-24z.5000samp.png)   |  ![](img/aa_cornell.2017-09-29_05-46-45z.5000samp.png)  |

You can definitely see the jagged edges are more profound on the left since AA is turned off. AA was achieved by randomly offsetting/jittering the pixel at each iteration.

## Materials

![](img/materials.png)

### Subsurface Scattering

![](img/subsurface/backleft.png)  |  ![](img/subsurface/backright.png) | ![](img/subsurface/frontright.png)
:-------------------------:|:-------------------------:|:-------------------------:

### Material Comparisons
Diffuse             |  Specular
:-------------------------:|:-------------------------:
![](img/diffuse1000.png)  |  ![](img/specular1000.png) |


Transmissive          |  Subsurface
:-------------------------:|:-------------------------:
![](img/transmissive1000.png)  |  ![](img/subsurface1000.png) 


## Integration/Shading Techniques

Naive             |  Direct Lighting
:-------------------------:|:-------------------------:
![](img/naive_cornell.2017-09-23_19-25-09z.1285samp.png)  |  ![](img/direct_cornell.2017-09-29_06-05-30z.5000samp.png)

There is no global illumination in direct lighting. When a ray hits an object in the scene, we sample its BSDF and shoot a second ray towards the light. We sample a random position on the light and determine its contribution. If there is another object in between the original object and the light, then the point is in shadow, otherwise the light is not occluded and we account for the light's contribution. Direct lighting only does a single bounce as opposed to the naive integrator.

In addition to not having Global Illumination, there is no support for transmissive and specular objects since there is only one bounce per iteration.

# Future Work

- Full Lighting/MIS
- Photon Mapping
- Acceleration Structures
- OBJ Loader

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