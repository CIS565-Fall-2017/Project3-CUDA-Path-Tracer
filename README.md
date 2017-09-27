**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 3 - CUDA Path Tracer**

* Josh Lawrence
* Tested on: Windows 10, i7-6700HQ @ 2.6GHz 8GB, GTX 960M 2GB  Personal

**/src CMakeLists.txt Additions**<br />
shapefunctions.h<br />
utilkern.cu<br />
warpfunctions.h<br />

**Overview**<br />
For a detailed overiew of the path tracing techniques see the book Physcially-Based Rendering:
http://www.pbrt.org/

**Highlights**<br />
<br />
<br />

**MIS with Fresnel Reflection and Transmission**<br />
![](img/cornellFresnelReflectionAndTransmissionMIS5000.png)

**MIS VS Naive 100spp**<br />
![](img/cornellMISvsNAIVE100.png)

**Data: Caching First Hit**<br />
**Data: Sorting by Material**<br />

**GPU Device Properties**<br />
https://devblogs.nvidia.com/parallelforall/5-things-you-should-know-about-new-maxwell-gpu-architecture/<br />
cuda cores 640<br />
mem bandwidth 86.4 GB/s<br />
L2 cache size 2MB<br />
num banks in shared memory 32<br />
number of multiprocessor 5<br />
max blocks per multiprocessor 32<br />
total shared mem per block 49152 bytes<br />
total shared mem per MP 65536 bytes<br />
total regs per block and MP 65536<br />
max threads per block 1024<br />
max threads per mp 2048<br />
total const memory 65536<br />
max reg per thread 255<br />
max concurrent warps 64<br />
total global mem 2G<br />
<br />
max dims for block 1024 1024 64<br />
max dims for a grid 2,147,483,647 65536 65536<br />
clock rate 1,097,5000<br />
texture alignment 512<br />
concurrent copy and execution yes<br />
major.minor 5.0<br />
