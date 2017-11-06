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

		void scan_implementation(int n, int *odata, const int *idata) 
		{
			// TODO
			int sum = 0;
			for (int i = 0; i < n; ++i) {
				odata[i] = sum;
				sum += idata[i];
			}
		}

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata)
		{
	        timer().startCpuTimer();
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
			int num = 0;
			for (int i = 0; i < n; ++i) {
				int val = idata[i];
				if (val != 0) {
					odata[num] = val;
					num++;
				}
			}

	        timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        
	        // TODO

			int* temp = new int[n + 1];
			temp[n] = 0;
			int num = 0;
			timer().startCpuTimer();
			// compute temporary array
			for (int i = 0; i < n; ++i) {
				temp[i] = (idata[i] == 0) ? 0 : 1;
			}
			// run exclusive scan on temporary array
			int sum = 0;
			for (int i = 0; i <= n; ++i) {
				int val = temp[i];
				temp[i] = sum;
				sum += val;
			}

			// scatter
			for (int i = 1; i <= n; ++i) {
				if (temp[i] != temp[i-1]) {
					odata[num] = idata[i - 1];
					num++;
				}
			}
			timer().endCpuTimer();
			delete[] temp;
            return num;
        }

        int findMax(int n, const int *idata) 
        {
        	int max = idata[0];
        	for (int i = 1; i < n; ++i) {
        		int val = idata[i];
        		if (val > max) {
        			max = val;
        		}
        	}

			return max;
        }

        void countSort(int n, int d, int *odata, int *idata)
        {
        	int count[10] = {0};

        	for (int i = 0; i < n; ++i) {
        		count[(idata[i]/d)%10] ++;
        	}

        	for (int i = 1; i < 10; ++i) {
        		count[i] += count[i - 1];
        	}

        	for (int i = n - 1; i >= 0; --i) {
        		odata[ count[(idata[i]/d)%10] - 1] = idata[i];
        		count[(idata[i]/d)%10] --;
        	}
        }

		/**
         * CPU sort
         */
        void sort(int n, int *odata, const int *idata)
		{
	        // TODO (from GeeksforGeeks Radix Sort)
			int* temp = new int[n];

			for (int i = 0; i < n; ++i) {
				odata[i] = idata[i];
			}
			
			int max = findMax(n, idata);

			timer().startCpuTimer();
			bool eo = true;
			for (int d = 1; max / d > 0; d *= 10) {
				countSort(n, d, temp, odata);
				std::swap(temp, odata);
				eo = !eo;
			}
			timer().endCpuTimer();

			if (!eo) {
				std::swap(temp, odata);
				std::memcpy(odata, temp, n * sizeof(int));
			}
			delete[] temp;
        }
    }
}
