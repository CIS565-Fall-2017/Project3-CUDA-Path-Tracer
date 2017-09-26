#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

		__global__ void computeNaiveScanHelper(int n, int d, int *odata, const int *idata);

        void scan(int n, int *odata, const int *idata);
    }
}
