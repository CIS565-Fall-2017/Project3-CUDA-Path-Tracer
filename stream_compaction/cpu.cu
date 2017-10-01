#include <cstdio>
#include "cpu.h"

#include "common.h"
#define TEST_TIME_IN_SCAN 0

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata, bool internalUse) {
          if (!internalUse) {
            timer().startCpuTimer();
          }

          odata[0] = 0;
          for (int i = 1; i < n; ++i) {
            odata[i] = idata[i - 1] + odata[i - 1];
          }
          if (!internalUse) {
            timer().endCpuTimer();
          }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
          int nextEmptyIdx = 0;
	        timer().startCpuTimer();
          for (int i = 0; i < n; ++i) {
            if (idata[i]) {
              odata[nextEmptyIdx++] = idata[i];
            }
          }
	        timer().endCpuTimer();
            return nextEmptyIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
          int *temp = (int*) malloc(n * sizeof(int));
          int *scannedTemp = (int*)malloc(n * sizeof(int));
	        timer().startCpuTimer();
	        // calculate temp array
          for (int i = 0; i < n; ++i) {
            temp[i] = (idata[i]) ? 1 : 0;
          }
          // scan
          scan(n, scannedTemp, temp, true);
          // scatter
          for (int i = 0; i < n; ++i) {
            if (temp[i]) {
              odata[scannedTemp[i]] = idata[i];
            }
          }
	        timer().endCpuTimer();
            return scannedTemp[n - 1] + (temp[n - 1] ? 1 : 0);
        }

        /**
        * CPU sort function.
        *
        * @returns the number of elements remaining after compaction.
        */
        void sort(int n, int *odata, const int *idata) {
          memcpy(odata, idata, n * sizeof(int));
          timer().startCpuTimer();
          std::sort(odata, odata + n);
          timer().endCpuTimer();
        }
    }
}
