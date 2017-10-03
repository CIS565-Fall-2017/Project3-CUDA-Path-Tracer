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
			int sum = 0;
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				sum += idata[i-1];
				odata[i] = sum;
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

		int compactWithoutScanSegment(int n, PathSegment *odata, const PathSegment *idata) {
			//timer().startCpuTimer();
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i].remainingBounces != 0) {
					odata[count] = idata[i];
					count++;
				}
			}
			//timer().endCpuTimer();
			return count;
		}

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			int *tempArray = new int[n];
			int *scanArray = new int[n];
	        //timer().startCpuTimer();
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					tempArray[i] = 1;
				}
				else {
					tempArray[i] = 0;
				}
			}
			scan(n, scanArray, tempArray);
			int count = 0;
			for (int i = 0; i < n-1; i++) {
				if (tempArray[i] == 1) {
					odata[scanArray[i]] = idata[i];
					count++;
				}
			}
	        //timer().endCpuTimer();
			delete[] tempArray;
			delete[] scanArray;
            return count;
        }
    }
}
