#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
	    StreamCompaction::Common::PerformanceTimer& timer();

		void radixSort(const int n, const int numbits, int *odata);

        void scan(const int n, int *odata, const int *idata);

        int compactWithoutScan(const int n, int *odata, const int *idata);

        int compactWithScan(const int n, int *odata, const int *idata);
    }
}
