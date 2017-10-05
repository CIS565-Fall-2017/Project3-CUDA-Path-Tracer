set(SOURCE_FILES
    "stb.cpp"
    "misHelper.h"
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
    "tiny_obj_loader.cc"
    "tiny_obj_loader.h"
    "preview.h"
    "preview.cpp"
    "utilities.cpp"
    "utilities.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
