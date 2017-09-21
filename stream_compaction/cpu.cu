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
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          
          if (n <= 0) {
            return;
          }

          odata[0] = 0;
          for (int i = 1; i < n; i++) {
            odata[i] = odata[i - 1] + idata[i - 1];
          }
	        
          timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
			    timer().startCpuTimer();
          
          int count = 0;
          for (int i = 0; i < n; i++) {
            if (idata[i] != 0) {
              odata[count] = idata[i];
              count++;
            }
          }

          timer().endCpuTimer();
          return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        
          // Map the input array to array of 0s and 1s
          int* arr1 = new int[n];
          int* arr2 = new int[n];
          for (int i = 0; i < n; i++) {
            if (idata[i] != 0) {
              arr1[i] = 1;
            }
            else {
              arr1[i] = 0;
            }
          }

          // Scan
          scan(n, arr2, arr1);

          // Scatter
          int count = 0;
          for (int i = 0; i < n; i++) {
            if (arr1[i]) {
              odata[arr2[i]] = idata[i];
              count++;
            }
          }

          delete[] arr1;
          delete[] arr2;

          timer().endCpuTimer();
          return count;
        }
    }
}
