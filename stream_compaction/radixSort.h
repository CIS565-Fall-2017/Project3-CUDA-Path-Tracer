#pragma once

#include "common.h"

namespace StreamCompaction {
  namespace RadixSort {
    StreamCompaction::Common::PerformanceTimer& timer();

    void radixSort(int n, int *odata, const int *idata, bool ascending);

  }
}
