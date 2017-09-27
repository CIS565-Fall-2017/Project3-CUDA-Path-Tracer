set(SOURCE_FILES
    "tiny_obj_loader.h"
    "tiny_obj_loader.cc"
    )

cuda_add_library(tinyobj
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )