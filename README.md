CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ricky Rajani
* Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, NVIDIA Quadro K620 (Moore 100C Lab)

// TODO: Explain project

// TODO: Add screenshots with labels and number of iterations
// Open with basic features and anti-aliasing
// Closed with basic features and anti-aliasing
// DOF with anti-aliasing, stream compaction, 
// Motion Blur with anti-aliasing
// Refractive with anti-aliasing
// Reflective with anti aliasing

// TODO: Add features implemented

### Performance Analysis

Stream compaction helps most after a few bounces. Print and plot the effects of stream compaction within a single iteration (i.e. the number of unterminated rays after each bounce) and evaluate the benefits you get from stream compaction.

Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?

For optimizations that target specific kernels, we recommend using stacked bar graphs to convey total execution time and improvements in individual kernels. For example:

