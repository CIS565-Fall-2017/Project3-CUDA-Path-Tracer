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
            // TODO

			// Exclusive scan
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
            // TODO

			int idx = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[idx] = idata[i];
					idx++;
				}
			}
			
	        timer().endCpuTimer();
            return idx;
        }

		void exclusiveScan(int n, int *odata, const int *idata) {
			// Exclusive scan
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			int *temp = new int[n];
			int *scanned = new int[n];

			timer().startCpuTimer();
	        // TODO

			// Create a temporary array 
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) {
					temp[i] = 0;
				}
				else {
					temp[i] = 1;
				}
			}

			// Exclusive Scan
			exclusiveScan(n, scanned, temp);

			// Scatter
			int numElements = 0;
			for (int i = 0; i < n; i++) {
				if (temp[i] == 1) {
					odata[scanned[i]] = idata[i];
					numElements++;
				}
			}

			timer().endCpuTimer();
			delete[]temp;
			delete[]scanned;

			return numElements;
        }
    }
}
