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

		void scan_implement(int n, int *odata, const int *idata) {
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}

        /**
         * CPU scan (prefix sum).
		 * Exclusive prefix sum
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
			scan_implement(n, odata, idata);
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
	        // TODO
			int* tempdata = new int[n];
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					tempdata[i] = 1;
				}
				else {
					tempdata[i] = 0;
				}
			}
			int* tempIndexdata = new int[n];
			scan_implement(n, tempIndexdata, tempdata);
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (tempdata[i]) {
					odata[tempIndexdata[i]] = idata[i];
					count++;
				}
			}
	        timer().endCpuTimer();
            return count;
        }
    }
}
