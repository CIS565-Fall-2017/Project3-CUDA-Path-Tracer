set(SOURCE_FILES
    "bvh.h"
    "stb.cpp"
    "image.cpp"
    "image.h"
    "interactions.h"
    "intersections.h"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "pathtrace.cu"
    "pathtrace.h"
    "scene.cpp"
    "scene.h"
    "sceneStructs.h"
    "preview.h"
    "preview.cpp"
    "tiny_obj_loader.h"
    "tiny_obj_loader.cc"
    "utilities.cpp"
    "utilities.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )


LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")