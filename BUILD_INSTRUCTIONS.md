Proj 3 CUDA Path Tracer - Build Instructions
========================

## The following are build instructions for a windows machine 

## Part 1: Setting up your development environment

1. Make sure you are running Windows 7/8/10 and that your NVIDIA drivers are
   up-to-date. You will need support for OpenGL 4.0 or better to run this project.
2. Install Visual Studio 2015.
   * 2012/2013 will also work, if you already have one installed.
   * 2010 doesn't work because glfw only supports 32-bit binaries for vc2010.
   **libraries not present for Win32**
   * You need C++ support. None of the optional components are necessary.
3. Install [CUDA 8](https://developer.nvidia.com/cuda-downloads).    
   * Use the Express installation. If using Custom, make sure you select Nsight for Visual Studio.
4. Install [CMake](http://www.cmake.org/download/). (Windows binaries are under "Binary distributions."")
5. Install [Git](https://git-scm.com/download/win).

## Part 2: Fork & Clone

1. Use GitHub to fork this repository into your own GitHub account.
2. If you haven't used Git, you'll need to set up a few things.
   * On Windows: In order to use Git commands, you can use Git Bash. You can
     right-click in a folder and open Git Bash there.
3. Clone from GitHub onto your machine:
   * Navigate to the directory where you want to keep your 565 projects, then
     clone your fork.
     * `git clone` the clone URL from your GitHub fork homepage.

* [How to use GitHub](https://guides.github.com/activities/hello-world/)
* [How to use Git](http://git-scm.com/docs/gittutorial)

## Part 3: Build & Run

* `src/` contains the source code.
* `external/` contains the binaries and headers for GLEW and GLFW.

### Windows

1. In Git Bash, navigate to your cloned project directory.
2. Create a `build` directory: `mkdir build`
   * (This "out-of-source" build makes it easy to delete the `build` directory
     and try again if something goes wrong with the configuration.)
3. Navigate into that directory: `cd build`
4. Open the CMake GUI to configure the project:
   * `cmake-gui ..` or `"C:\Program Files (x86)\cmake\bin\cmake-gui.exe" ..`
     * Don't forget the `..` part!
   * Click *Configure*.  Select your version of Visual Studio, Win64.
     (**NOTE:** you must use Win64, as libraries aren't provided for Win32.)
   * If you see an error like `CUDA_SDK_ROOT_DIR-NOTFOUND`,
     set `CUDA_SDK_ROOT_DIR` to your CUDA install path. This will be something
     like: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0`
   * Click *Generate*.
5. If generation was successful, there should now be a Visual Studio solution
   (`.sln`) file in the `build` directory that you just created. Open this.
   (from the command line: `explorer *.sln`)
6. Build. (Note that there are Debug and Release configuration options.)
7. Run. Make sure you run the `cis565_` target (not `ALL_BUILD`) by
   right-clicking it and selecting "Set as StartUp Project".
   * If you have switchable graphics (NVIDIA Optimus), you may need to force
     your program to run with only the NVIDIA card. In NVIDIA Control Panel,
     under "Manage 3D Settings," set "Multi-display/Mixed GPU acceleration"
     to "Single display performance mode".