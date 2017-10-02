CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yi Guo
* Tested on:  Windows 8.1, Intel(R) Core(TM)i5-4200M CPU @ 2.50GHz 8GB, NVIDIA GeForce 840M (Personal Notebook)

### (TODO: Your README)

## Description
     
This is a path tracing rendering system implemented on GPU. We use the parallel algorithm of reduction,scan, stream compaction and radix sort to manage a ray tracer pool. For the instructions of the project, see `INSTRUCTION.md`. Here are the images and performance analysis of the project.
           
## Screenshot 
         
![](./images/mini_demo.gif)
      
## Performance Analysis
        
* **Stream Compacting of the Ray Tracers**
On cpu, we will use multiple threads to deal with multiple pixels at the same time. But on Gpu, since the time costs for different pixels are different (which is the divergence) and threads can be in flight for a long time, it is a enormous waste for the computational resource (If some threads are tracing more bounces and some are only tracing a few, this will end up with a lot of idling threads.).So instead, we use one thread to deal with one ray at a time and remove the terminated ray from the pool. The size of the ray pool will probably get smaller and smaller with each iteration and each iteration should generally execute faster than previous. 

* **Stream Compacting of the Ray Tracers in the closed scene**
The rays in a closed scene will probably bonuce more times than the rays in the open scene, since in the closed scene, the rays are less likely to hit nothing. Also, since almost each ray will return a meaningful value instead of 0, the overall illumination will be. 

Here are the plots showing the changes of the number of rays in the pool and the time cost for each depth in a open scene and a closed scene.

![](./images/RayNum_Open_vs_Close.png)
        
![](./images/timecost_Open_vs_Close.png) 

As the plot shows, in the open scene, the number of rays in the pool will decrease dramatically, while in the closed scene, the changes is not very obvious. That makes sense because in the closed scene, the rays have less possibility to hit nothing and most of them will keep bouncing untill the depth reaches the limitation.

Open Scene
![](./images/OpenGlass.png) 
      
Closed Scene
![](./images/CloseGlass.png) 

## Extra features
* **Efficient material sort**
When the rays in different threads hit the object with different material, the threads will get into different branches and the overall efficency will be decreased. To reduce the branches of the tracing process, we need to sort the ray according to their material so that the rays in one block will probably hit the same material. However, since our scene is very simple and there are not too many different materials for in a row(we scan the pixel row by row), sorting by material may not produce great difference.

But we still need to talk about the way of sorting by materials, because it will be necessary when the scene become complex. There are mainly 2 ways for sorting the ray by materials. One is sorting the ray directly and the other is sorting the indices of the ray. The comparison of the efficency of 2 methods is shown below(Method 1 is sorting by indices and method 2 is sorting directly by ray). Obviously, sorting the indices will be more than 3 times faster than directly sorting the ray. That is because the `Ray` here is a struct or class obj and its size is much bigger than a simple integer.Moreover, it is saved on the global memory of GPU and accessing such a huge data will take much more time than accessing a simple integer. 

![](./images/Sorting.png) 

* **MIS and Direct Lighting**
For the naive ray tracing method, we cannot get a very good illumination distribution since many rays will hit nothing and return 0 value. We want the rays have higher possibility to hit the light, because the light contribute most for the global illumination(which is the idea of importance sampling). For MIS, See PBRT for more info. By using MIS, we get a image with better illumination distribution in less iterations.  

Naive 200 iterations
![](./images/cornellNaive.png) 

Naive 200 iterations
![](./images/cornellMIS.png) 

* **Russian roulette**
Russian roulette will terminated the rays that have very little energy and directly compute the compensation for that part, which will decrease the times of iterations and improve the overall efficency. As the graph shows, Russian roulette will produce great effect when the scene is closed.

Open Scene
![](./images/Russian_roulette_Open.png) 

Closed Scene
![](./images/Russian_roulette_Close.png) 

* **Other features**
Some other features, like depth field camera, glass(refraction) material ,are also included in the project. Here are the images.

       
Normal camera
![](./images/cornellNoDepth.png) 
      
Depth field camera
![](./images/cornellDepthCam.png) 


* **BVH Tree**
Still updating......