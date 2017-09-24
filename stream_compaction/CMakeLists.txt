set(SOURCE_FILES
    "EfficientStreamCompaction.cu"
    "EfficientStreamCompaction.h"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
