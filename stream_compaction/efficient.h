#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		__global__ void upSweep(int n, int factorPlusOne, int factor, int addTimes, int *idata);

		__global__ void downSweep(int n, int factorPlusOne, int factor, int addTimes, int *idata);

		__global__ void resizeArray(int n, int new_n, int *idata);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
