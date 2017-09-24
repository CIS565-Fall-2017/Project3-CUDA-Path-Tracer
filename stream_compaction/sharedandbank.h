
#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedAndBank {
        StreamCompaction::Common::PerformanceTimer& timer();

		__global__ void kernZeroExcessLeaves(const int pow2roundedsize, const int orig_size, int* data);
		__global__ void kernScan(const int shMemEntries, int* idata, int* SUMS);
		__global__ void kernAddBack(const int n, int* idata, const int* scannedSumsLevel);
		void recursiveScan(const int n, const int level, int *idata);
        void scan(const int n, int *odata, const int *idata);
        void scanNoMalloc(const int n, int *dev_idata);
        int compact(const int n, int *odata, const int *idata);
        int compactNoMalloc(const int n, int *idata);
    }
}