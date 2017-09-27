#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

		void scanExcusivePrefixSum(int N, int dimension, dim3 fullBlocksPerGrid, dim3 threadsPerBlock, int *dev_oDataArray);
    }
}
