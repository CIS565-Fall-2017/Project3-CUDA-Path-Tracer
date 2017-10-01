CUDA Path Tracer
================
![](./results/title.jpg)
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Jiawei Wang
* Tested on: Windows 10, i7-6700 @ 2.60GHz 16.0GB, GTX 970M 3072MB (Personal)
## Results:
___
![](./results/010.gif)
### Basic Path Tracer:
![](./results/cornell.2017-09-24_15-39-02z.5000samp.png "5000 samples/ cornell")
### Other Features:
* **Anti-Aliasing**: Realized by Stochastic *Sampled Antialiasing*, just jitter a little bit when create Camera ray, this will cause a little blur on the image, which can solve the aliasing problems.

  Here is the results: (The left image is without anti-aliasing, the right one is with anti-aliasing)
<img src="./results/cornell.2017-09-24_21-33-57z.5000samp_no_antialiasing.png" width="400" height="400"> <img src="./results/cornell.2017-09-25_21-30-06z.5000samp_with_antialiasing.png" width="400" height="400">
<img src="./results/without_anti.jpg" width="400" height="400"> <img src="./results/with_anti.jpg" width="400" height="400">


