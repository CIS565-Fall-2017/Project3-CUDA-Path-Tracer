#include <cstdio>
#include "cpu.h"

#include "common.h"

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
        void scan_implementation(int n, int *odata, const int *idata) {
          // your actual implementation
          odata[0] = 0;
          for (int i = 1; i < n; i++) {
            odata[i] = odata[i - 1] + idata[i - 1];
          }
        }

        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          // TODO
          scan_implementation(n, odata, idata);
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          // TODO
          int next = 0;
          for (int i = 0; i < n; i++) {
            if (idata[i] != 0) {
              odata[next] = idata[i];
              next++;
            }
          }
	        timer().endCpuTimer();
          return next;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
          int *mapped = new int[n];
          int *scanned = new int[n];
          int count = 0;
          // Map
          for (int i = 0; i < n; i++) {
            if (idata[i] != 0) {
              mapped[i] = 1;
            }
            else {
              mapped[i] = 0;
            }
          }

          // Scan
          scan_implementation(n, scanned, mapped);

          // Scatter
          for (int i = 0; i < n; i++) {
            if (mapped[i] == 1) {
              odata[scanned[i]] = idata[i];
              count++;
            }
          }

	        timer().endCpuTimer();
          return count;
        }
    }
}
