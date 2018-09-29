set(SOURCE_FILES
	"common.h"
	"efficient.h"
	"common.cu"
	"efficient.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_50
    )
