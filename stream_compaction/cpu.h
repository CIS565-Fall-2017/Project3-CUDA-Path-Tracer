#pragma once

#include "common.h"
#include "../src/sceneStructs.h"

namespace StreamCompaction {
    namespace CPU {
	    StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

		int compactWithoutScanSegment(int n, PathSegment *odata, const PathSegment *idata);

        int compactWithScan(int n, int *odata, const int *idata);
    }
}
