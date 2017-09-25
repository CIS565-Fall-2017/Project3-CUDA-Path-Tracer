#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
		
		//Intended to compact in place on the gpu
		int compactOnGpu(int n, int *odata, const int *idata);
    }
}
