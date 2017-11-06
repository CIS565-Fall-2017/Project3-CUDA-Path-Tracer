#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        void sort(int n, int d, int *odata, const int *idata);
    }
}
