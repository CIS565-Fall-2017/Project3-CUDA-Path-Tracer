#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		void non_opt_scan(int n, int *odata, const int *idata);

		__global__ void cudaSweepUp(int n, int d, int *data);

		__global__ void cudaSweepDown(int n, int d, int *data);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
