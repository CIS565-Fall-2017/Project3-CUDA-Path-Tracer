CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yuxin Hu
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### Yuxin Hu
* Debugging Process
  * Error 1. Insufficient Space Allocation for Stream Compaction boolean and indice arrays
  ![Insufficient Space Allocation for Stream Compaction](/img/cornell.debug.InsufficientStreamCompactionSpace.50samp.png)
  <p align="center"><b>Insufficient Space Allocation for Stream Compaction</b></p>
  
  I was using the stream compaction I wrote in last project to remove terminated rays from ray pool. However the stream compaction I wrote last time assumes the boolean and indices arrays allocated on GPU has a size that is power of 2. For this project, the default image is 800*800, which initializes a pool rays with 640000 threads, and it is not a power of 2 number. When I initialized the and allocated space for boolean and indices array on GPU for stream compaction, I allocated 640000 * size(dataType) for both arrays, which will have a an error during the first round of stream compaction in efficient scan process. This will result the rays in the upper part of image being trimmed off for subsequent tracing.
  
  * Error 2. Did not offset the origin for new ray.
  ![No Offset on New Ray Origin](/img/cornell.debug.pureDiffusive.png)
  <p align="center"><b>No Offset on New Ray Origin</b></p>
  
  I did not give an offset for the new ray origin along the new ray direction, then I produced above image.

