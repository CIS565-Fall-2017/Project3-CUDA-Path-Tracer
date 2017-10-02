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
        void scan(int n, int *odata, const int *idata, bool timer_on) {
			if(timer_on)
				timer().startCpuTimer();
            // TODO
			/*int test[8] = { 3,1,7,0,4,1,6,3 };
			int test_results[8] = {};

			test_results[0] = 0;
			for (int k = 1; k < 8; k++) {
				test_results[k] = test_results[k - 1] + test[k - 1];
			}
		*/
			if (n <= 0)
				return;
			odata[0] = 0;
			for (int k = 1; k < n; k++) {
				odata[k] = odata[k - 1] + idata[k - 1];
			}
			if(timer_on)
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
			if (n <= 0)
				return 0;
			int count = 0;
			for (int k = 0; k < n; k++) {
				if (idata[k])
					odata[count++] = idata[k];
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
			int *check_array = new int[n];
			for (int k = 0; k < n; k++) {
				if (idata[k])
					check_array[k] = 1;
				else
					check_array[k] = 0;
			}
			scan(n, odata, check_array, false);
			delete check_array;
			int count = odata[n - 1] + idata[n - 1];
			for (int k = 0; k < n; k++) {
				if (idata[k])
					odata[odata[k]] = idata[k];
			}
	        timer().endCpuTimer();
            return count;
        }
    }
}
