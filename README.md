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
    Below are some are some renders using multiple importance sampling and and naive path tracing(no direct light sampling). Notice the noise rection due to direct light importance sampling in the 100 samples per pixel image. 
<br />
<br />
    The compaction optimization was not much of an optimization given that it did not reduce the render time per pixel sample. Reasons for this could be the sorting overhead. I tried to increase the pixel count to 1000x1000 image and did not see the compaction fuctions catching up to no compaction version. The warp retiring benefits we get from removing the dead paths does not outway the overhead from removing them in the first place.
<br />
<br />
    Of three optimizations that were performed, first bounce caching was the only one that provided any performance improvement. Across several depth termination settings it provided a constant savings of roughly 2ms. This is the amount of time it takes to determine the first itersection from the camera. 
<br />
<br />
    Material sorting was much worse than not sorting at all. This is likely due to the overhead of sorting path segments by material type. Also, the purported benefits of sorting paths by material type was that if one type of material had many instructions to complete compared to other materials then you would get some performance benefit by only taking that code path for the warp. However, these performance benefits are only realized if the intersections per material are mutiples of 32, allowing the warps to execute one material code path. This is highly unlikely. It is far more likely they are not multiples of 32 and some warps must stride 2 or more materials. This negates the benefits of sorting becuase in order for the next kernel call to start executing, it must wait around for these kinds of mixed material warps to finish. The only way I could see this working with existing code is separating the uber shader up by material and kicking off each kernel in its own stream.
<br />
<br />
    The reasons are as follows: the uber kernel should take C + sum(Mi) instructions to complete where C is some constant instruction overhead for all the material types and Mi is the number of instructions it takes to execute a specific material shading code path for material i. The separated kernel approach with different kernel streams per material should only take C + max(Mi) where max Mi is the longest material code path. A savings of sum(Mi) - max(Mi). If this is not more than the cost to sort then there no reason to sort.
<br />
<br />


**MIS with Fresnel Reflection and Transmission**<br />
**800x800 5000spp**<br />
![](img/cornellFresnelReflectionAndTransmissionMIS5000.png)

**800x800 5000spp**<br />
![](img/cornellCubeFresnelReflectionAndTransmissionMIS5000.png)

**MIS VS Naive**<br />
**800x800 100spp**<br/>
![](img/cornellMISvsNAIVE100.png)

**Data**<br />
![](img/data.png)

**Compacting the non-dead Paths**<br />
![](img/compact.png)

**Caching the First Bounce**<br />
![](img/firstbouncecaching.png)

**Material Sorting**<br />
![](img/materialsorting.png)


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
