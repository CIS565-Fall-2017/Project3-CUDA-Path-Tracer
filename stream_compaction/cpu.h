#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
	    StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

		void scanExclusivePrefixSum(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);

		int scatter(int n, int *mapped, int *scanned, const int *idata, int *odata);

		void map(int n, int *mapped, const int *idata);
    }
}
