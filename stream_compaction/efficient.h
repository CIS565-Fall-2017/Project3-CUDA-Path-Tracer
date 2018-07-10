#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		__global__ void kernZeroExcessLeaves(const int pow2roundedsize, const int orig_size, int* data);
		__global__ void kernScanDown(const int pow2roundedsize, const int indexscaling, const int offset, int* data);
		__global__ void kernScanUp(const int pow2roundedsize, const int indexscaling, const int offset, int* data);

        void scan(const int n, int *odata, const int *idata);
        void scan_notimer(const int n, int *odata, const int *idata);

        int compact(const int n, int *odata, const int *idata);
    }
}
