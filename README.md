CUDA Path Tracer
================
![](./results/title.jpg)
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Jiawei Wang
* Tested on: Windows 10, i7-6700 @ 2.60GHz 16.0GB, GTX 970M 3072MB (Personal)
## Results:
___
![](./results/010.gif)

### Final Path Tracer:
![](./results/test1.2017-10-01_06-21-18z.3507samp.png "5000 samples/ cornell")
### Detailed Features:
* **Anti-Aliasing**: Realized by Stochastic *Sampled Antialiasing*, just jitter a little bit when create Camera ray, this will cause a little blur on the image, which can solve the aliasing problems.</br>
  Here is the results: (left: `without anti-aliasing`, right: `with anti-aliasing`)
<img src="./results/cornell.2017-09-24_21-33-57z.5000samp_no_antialiasing.png" width="400" height="400"> <img src="./results/cornell.2017-09-25_21-30-06z.5000samp_with_antialiasing.png" width="400" height="400"></br>
<img src="./results/without_anti.JPG" width="400" height="250"> <img src="./results/with_anti.JPG" width="400" height="250"></br>

* **Depth Of Field**: Realized by jittering around pixel. See at [PBRT 6.2.3]</br>
  Here is the results: </br>
  (left: `lens_radius = 0.5`, right: `lens_radius = 1.0`)</br>
<img src="./results/cornell.2017-09-27_23-22-31z.5000samp.png" width="400" height="400"> <img src="./results/cornell.2017-09-27_23-00-16z.5000samp.png" width="400" height="400"></br>
  (left: `focal_length = 9`, right: `focal_length = 19`)</br>
<img src="./results/test1.2017-09-28_00-41-30z.5000samp.png" width="400" height="400"> <img src="./results/test1.2017-09-28_00-50-21z.5000samp.png" width="400" height="400"></br>

* **Refraction(Frensel Effect)**: Frensel Effect requires not only the refration effect, but also the reflection effect. This is realized by generate a random value for each sample and use this value comparing with the **dot product** to decide whether this sample is counted as a reflection sample.</br>
  Here is the results: (left: `no reflection`, right: `both reflection and refraction`)</br>
<img src="./results/cornell.2017-09-27_19-52-54z.5000samp_without_reflection.png" width="400" height="400"> <img src="./results/cornell.2017-09-27_20-31-09z.5000samp_with_reflection.png" width="400" height="400"></br>
  You can see that the right one is more realistic, 'cause the ball reflects the light on the celling.</br>
  
* **Motion blur**: Realized by jittering ray of different iterations between the `transition_start` to the `transition_end`, which can generate an effect like followings:</br>
<img src="./results/cornell.2017-09-26_03-33-04z.5000samp.png" width="400" height="400"> <img src="./results/cornell.2017-09-26_04-39-45z.5000samp.png" width="400" height="400"></br>

* **Direct light**: In this project, I only implemented the most basic direct light, make the ray direction to the ligths when the `ray.remainning bounce == 1`, and then average the color of them. </br>
  Here is the results: (left: `no direct light`, right: `direct light`) </br>
<img src="./results/cornell.2017-09-28_00-12-02z.5000samp.png" width="400" height="400"> <img src="./results/direct_light_test.png" width="400" height="400"> </br>

* **Arbitrary Mesh Loading**: Used `tiny_obj` tools to load the obj file and `glm::intersectRayTriangle` function to do the intersection test.</br>
  Here is the results: </br>
<img src="./results/test1.2017-10-01_06-21-18z.3507samp.png" width="400" height="400"> <img src="./results/cornell.2017-09-30_18-54-18z.5000samp.png" width="400" height="400"> </br>
  


