#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();
// scanShared does the same scan but now it uses shared memory
// and does the scan in blocks
        void scanShared(int n, int *odata, const int *idata, 
			int tileSize = -1);
        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
