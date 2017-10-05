#include <cstdio>
#include "cpu.h"
#include <thread>

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
			try {
				timer().startCpuTimer();
			}
			catch (...){};
			/* FAILED ATTEMPT AT GPU WAY
			//Copy idata into odata
			for (int i = 0; i < n; i++) {
				odata[i] = idata[i];
			}

			// Create Constants
			const int logN = ilog2ceil(n);

			printf("LOG N IS : %d \n", logN);
			
			//Up-Sweep
			for (int d = 0; d < logN; d++) {
				for (int k = 0; k < n; k += (int)pow(2, d + 1)) {
					odata[k + (int)pow(2, d + 1) - 1] += odata[(k + (int)pow(2, d) - 1)];
				}
			}

			printf("UPSWEPT: \n (");
			for (int i = 0; i < 15; i++) {
				printf("%d, ", odata[i]);
			}
			printf(".... )\n");

			//Down-Sweep
			odata[n - 1] = 0;
			for (int d = logN-1; d >= 0; d--) {
				for (int k = 0; k < n; k += (int)pow(2, d + 1)) {
					if ((k + (int)pow(2, d+1) - 1) < n) {
						int t = odata[(k + (int)pow(2, d) - 1)];
						odata[k + (int)pow(2, d) - 1] = odata[k + (int)pow(2, d + 1) - 1];
						odata[k + (int)pow(2, d + 1) - 1] += t;
					}
				}
			}

			printf("SUMMED: \n (");
			for (int i = 0; i < 15; i++) {
				printf("%d, ", odata[i]);
			}
			printf(".... )\n");

			*/
			odata[0] = 0;
			for (int k = 1; k < n; k++) {
				odata[k] = idata[k-1] + odata[k-1];
			}

			try {
				timer().endCpuTimer();
			}
			catch (...) {};
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
					odata[count++] = idata[i];
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
			int count = 0;
			int* temp     = (int*) malloc(sizeof(int) * n);
			int* scanned  = (int*) malloc(sizeof(int) * n);

			for (int i = 0; i < n; i++) {
				temp[i] = (int) (idata[i] != 0);
				count = idata[i] != 0 ? count + 1 : count;
			}
			
			scan(n, scanned, temp);

			for (int i = 0; i < n; i++) {
				if (temp[i] == 1) {
					odata[scanned[i]] = idata[i];
				}
			}

			try {
				timer().endCpuTimer();
			}
			catch (...) {};

            return count;
        }
    }
}
