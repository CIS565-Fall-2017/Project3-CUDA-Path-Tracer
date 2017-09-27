CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ricky Rajani
* Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, NVIDIA Quadro K620 (Moore 100C Lab)

This project implements a CUDA-based path tracer capable of rendering globally-illuminated images very quickly. Base code was provided; hpwever, the core renderer was implemented by the author.

# Core Features:
- A shading kernel with BSDF evaluation for Ideal Diffuse surfaces [PBRT 8.3] and perfectly specular-reflective (mirrored) surfaces
- Path continuation/termination using Stream Compaction
- Path segments and intersections are contiguous in memory by material type before shading
- First bounce intersections are cached for re-use across all subsequent iterations

# Extra Features:
- Refraction [PBRT 8.2] with Frensel effects using Schlick's approximation
- Physically-based depth-of-field (by jittering rays within an aperture) [PBRT 6.2.3]
- Stochastic Sampled Antialiasing
- Motion blur through averaging samples at different times during the animation

*All features are easily toggleable

# Screenshots

Open scene with core features and anti-aliasing - 5000 iterations
![](img/samples/open-basic.PNG)

Closed scene with core features and anti-aliasing (only one light) - 5000 iterations
![](img/samples/closed-basic.PNG)

Open scene with depth-of-field, stream compaction and anti-aliasing - 5000 iterations
![](img/samples/dof.PNG)

![](img/samples/dof2.PNG)

Open scene with motion blur, stream compaction and anti-aliasing - 5000 iterations
![](img/samples/motion.PNG)

Open scene with refraction, core features and anti-aliasing - 5000 iterations
![](img/samples/refract.PNG)

### Performance Analysis

![](img/active-paths-graph.PNG)

Stream compaction helps most after a few bounces. It reduces the number of active paths during an iteration. The plot above shows the reduction in paths for open and closed scenes during one iteration. The thrust implementation (remove-if) was used for stream compaction as it was significantly faster than the work-efficient compaction.

Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?

Provide performance benefit analysis across different max ray depths for first bounce caching

Provide performance benefit analysis for sorting path segments and intersections

For optimizations that target specific kernels, we recommend using stacked bar graphs to convey total execution time and improvements in individual kernels.

### References
- [PBRT] Physically Based Rendering, Second Edition: From Theory to Implementation. Pharr, Matt and Humphreys, Greg. 2010.
