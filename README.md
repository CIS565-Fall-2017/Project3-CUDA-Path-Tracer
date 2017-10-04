# **University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3:**

# **CUDA Path Real Time Path Tracer**





Tested on: Windows 10, Intel Core i7-7700HQ CPU @ 2.80 GHz, 8GB RAM, NVidia GeForce GTX 1050

 ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/github/issues-raw/badges/shields/website.svg) ![CUDA 8.0](https://img.shields.io/badge/CUDA-8.0-green.svg?style=flat)  ![Platform](https://img.shields.io/badge/platform-Desktop-bcbcbc.svg)  ![Developer](https://img.shields.io/badge/Developer-Youssef%20Victor-0f97ff.svg?style=flat)




- [Features](#features)



- [Analysis](#analysis)



- [Observations](#observations)



- [Blooper](#blooper)
 

 

____________________________________________________


 
The goal of this project was to run an algorithm that clears out all zeros from an array on the GPU using CUDA. This parallel reduction is done using the scan algorithm that computes the exclusive prefix sum. I also implemented a parallelized Radix Sort using the exclusive prefix sum algorithm developed.



### Things Done

#### Core Features

 - [x] Shading for Ideal Diffuse Surfaces
 - [x] Shading for Perfectly Specular Surfaces
 - [x] Early Ray Termination Using Stream Compaction
 - [x] Caching of First Bounces

 #### Spicy Features
  - [x] Shading for Transmissive Surfaces (maybe)
  - [x] Work-efficient Stream Compaction Using Shared Memory Across Multiple Blocks
  - [x] Environment Maps
  - [x] Direct Lighting
  - [x] Multiple Importance Sampling
  - [x] Stochastic Antialiasing
  - [x] Arbitrary Mesh Loading
  - [ ] Kd-Tree & Stackless Kd-Tree Traversal on the GPU
  - [ ] [Physically Accurate Lens Flares](https://placeholderart.wordpress.com/2015/01/19/implementation-notes-physically-based-lens-flares/)
  - [ ] [Specular BRDF with Microstructure BRDF](https://people.eecs.berkeley.edu/~lingqi/publications/paper_glints2.pdf)
  - [ ] Texture Mapping (maybe)
  - [ ] Depth of Field/Cooler Lens Effects such as bloom
 
![Cornell Box](/img/cornellOBJ.png)





### Analysis:

All tests were run on a range of input from 2^7 (128) all the way to 2^15 (32768) sized arrays. This means that if my algorithms ran in linear O(n) time, then I should get a correlated graph that matches how fast my input increases, in other words, I should get an exponential graph such as this:

![O(n)](/images/correlation.png)

####Scan Algorithms

So for my scan algorithms I didn't initially see anything interesting about my results. It seemed that they were just randomly consistent. Here is what I got with all my scan algorithms, GPU and CPU, side-by-side:

![All of them](/images/scan-with-thrust.png)

Now you can see that there is a massive outlier: The Thrust scan with a power-of-two number of elements in the array. Since I didn't implement it, and it's skewing all the results and making them unreadable, let's take it out. We are left with this:


![All of them no thrust](/images/scan-without-thrust.png)

Now, we can still see some separation between the top results that were GPU-based, and the lower, CPU-based results. So let's start first with the latter.

*CPU Runtimes:*

![CPU Runtimes](/images/scan-cpu.png)

Disregarding the weird fluke in the end, I think it's obvious that this algorithm is more or less O(n). This makes perfect sense as my CPU algorithms run through the array only once. So I've established a baseline to beat.

Here is what it looks like with my GPU algorithms by themselves:

*GPU Runtimes:*

![GPU Runtimes](/images/scan-gpu.png)


Now, despite my GPU algorithms being slower than my CPU ones strictly in terms of runtime in milliseconds, it is clear that they scale much better than my CPU-based ones. This is the benefit of parallelization I'm assuming. The trendlines (dashed lines) show a clear linear correlation to my exponential input. This means that my GPU algorithms are better. Despite the bottleneck. (more on that in the [observations](#observations)).


####Compact Algorithms

With compact algorithms, we see the same trend continue. Here are the CPU algorithms:

![CPU Compact Runtimes](/images/compact-cpu.png)

and here is what the GPU Algorithms look like:

![GPU Compact Runtimes](/images/compact-gpu.png)

Again, we can see a linear GPU trendline, and an exponential trend with CPU. This is not a surprise as compact algorithms are based on the scan algorithms.

For reference, here is the CPU+GPU graph of compact algorithm runtimes:

![All Compact Runtimes](/images/compact.png)


---------------------------------


###Radix Sort Analysis

It took me a while to understand radix sort, but I finally figured it out. Debugging was really annoying as I didn't use any algorithms I found online as I found them too convoluted. They were also optimized but to an extent. They were all limited to small arrays, I wanted mine to scale up.

####How Parallel Radix Sort Works

Understanding it took a while but once I did, it was super easy to implement.

*Highest Level Approach:* Basically, it works like this: For each bit, from the least significant to the most significant, sort based on whether that bit is a zero or a one.

*Less High Approach:* The way you do this is by doing [key counting](https://en.wikipedia.org/wiki/Counting_sort). 

--- You start off by running an algorithm to tell you whether a given number has a `zero` or a `one` in that particular bit. This leaves you with your `histogram` array. For my purposes I created two. One for zeros and another for ones. So I had `hists0` and `hists1`. 

-- The next step is to run a exclusive prefix scan (that we already implemented) on the histograms array to get the "`offset`" of a number. This tells you where this number will be. Here, I also made two arrays, one for the `0` offsets, and one for the `1` offsets.

-- The final step is to place your numbers in the final array based on their offsets. This is done by adding `hists0` + `offsets0` and using that as your index. In code this would look kinda like this: `odata[hists0[i] + offsets0[i]] = idata[i]`

Now because this is fairly abstract, I made a visualization of the algorithm run on an array `a = [7, 3, 4, 11, 10]`. Feel free to use my explanation as a reference to help you understand.

*Visualized:*


![This animation took longer to make than it did to code](/images/radix.gif)

As far as runtimes go for this, they were actually closer to O(n), which is fairly accurate considering that is the runtime for radix sort. However, this should be faster. My guess is that this was because of the global memory that was being accessed back and forth. This also probably contributed to the increase in ~4ms around all array sizes. Here are the runtimes:

![Radix Sort Runtimes](/images/radix.png)


### My Block Size Optimization:

So I had a setup for an optimized running of this that I think would have helped a lot had I implemented shared memory. Basically the setup is as follows:

I have 96kb of Shared Memory per SM\*. That is roughly 24k ints allowed per SM. An optimally busy SM has 32 blocks per SM, which is the maximum for an SM\*. To make things even better you want to have it such that each thread only reads in one integer.

If we call the memory we access per thread `x` and the maximum allowed blocks per SM `z`, then we want to optimize the number of threads per block, which we define as `y`.

Put in equation form we want to have:

   `x * y * z <= 96000`

Now we have in this case `x = 4 bytes`, `z = 32 blocks`. And so we can rewrite our equation to be:

   `4 * y * 32 <= 96000`

We want `y` to be a multiple of 32 to match our warp sizes, and we also want it to be a bit less than `96000` as we probably use more memory per thread than exactly 4 bytes. I call this the "tone-down" value. I've gotten the optimal value for this by trial and error. I've found `0.85f` works best.


   `4 * y * 32 <= 96000*0.85f`

The nearest multiple of 32 that matches this is `608`. And so that is how I determine my block size\**. The number of blocks on the grid is just simply `n/block_size`

\* These numbers are based on my GTX 1050 (CC 6.1) I have them hardcoded into my code at the top of `common.h`

\** If an array is less than this "optimal block size" I just run everything on the same block since we definitely have enough shared memory then.

### Tests Raw Output:

```
****************
** SCAN TESTS **
****************
    [  15  10  42  24  29  20  33   5  10   5  16   2   0 ...  42   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.053971ms    (std::chrono Measured)
    [   0  15  25  67  91 120 140 173 178 188 193 209 211 ... 802690 802732 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.053606ms    (std::chrono Measured)
    [   0  15  25  67  91 120 140 173 178 188 193 209 211 ... 802656 802672 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.338944ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.300032ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.44544ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.493568ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 12.0873ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.230208ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   2   2   3   2   3   3   2   3   2   0   2 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.087885ms    (std::chrono Measured)
    [   1   2   2   2   3   2   3   3   2   3   2   2   3 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.086063ms    (std::chrono Measured)
    [   1   2   2   2   3   2   3   3   2   3   2   2   3 ...   2   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.158633ms    (std::chrono Measured)
    [   1   2   2   2   3   2   3   3   2   3   2   2   3 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.42496ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.41984ms    (CUDA Measured)
    passed

*****************************
** MY OWN RADIX SORT TESTS **
*****************************
==== radix basic, power-of-two ====

   elapsed time: 4.00589ms    (CUDA Measured)


==== radix basic, non power-of-two ====

   elapsed time: 3.56966ms    (CUDA Measured)

==== radix massive, power-of-two ====

   elapsed time: 4.9449ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 4.59571ms    (CUDA Measured)


*********** WITH 128 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 4.28851ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 4.08781ms    (CUDA Measured)


*********** WITH 256 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 4.35814ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 4.32435ms    (CUDA Measured)


*********** WITH 512 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 4.61926ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 4.46054ms    (CUDA Measured)


*********** WITH 1024 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 4.82816ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 4.66227ms    (CUDA Measured)


*********** WITH 2048 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 5.04115ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 4.8343ms    (CUDA Measured)


*********** WITH 4096 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 5.43437ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 4.87834ms    (CUDA Measured)


*********** WITH 8192 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 5.84192ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 5.13638ms    (CUDA Measured)


*********** WITH 16384 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 6.02829ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 5.45382ms    (CUDA Measured)


*********** WITH 32768 SIZED ARRAY *********
==== radix massive, power-of-two ====

   elapsed time: 6.46963ms    (CUDA Measured)


==== radix massive, non power-of-two ====

   elapsed time: 5.96787ms    (CUDA Measured)
```

### Observations:

* Bottlenecks:

My bottleneck was definitely memory access. This added a constant look up and runtime lag in my execution. This is further supported by my runtimes. All of my GPU code scaled linearly to an exponential increase in input which means that there is a constant delay in the time it takes to run the algorithm. This is the memory cycle as each global memory access took ~200 cycles as opposed to 2 cycles in shared memory and it's either 1 or 2 for CPU. which is what you see as the GPU runtimes are ~100-200 times slower. But they scale up linearly.



### Bloopers / Lessons Learned

Bloopers weren't that interesting to see this assignment, so here's another thing:

| **Lesson**                                                                                                                | **Cost of Learning**  |
|---------------------------------------------------------------------------------------------------------------------------|-----------------------|
| Apparently CUDA doesn't always tell you if you pass in a CPU Array by accident                                            | 7 hours               |
| If my gpu array is larger than "n", my algorithm will access things past n.  Despite any amount of checks I put in place. | 2 hours               |
| When iterating Radix Sort from 0 to MSB, remember off by one errors please.                                               | 3 hours maybe I dunno |

