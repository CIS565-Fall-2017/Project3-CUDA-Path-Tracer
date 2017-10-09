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


	//  occupied array converts an array of integers to an array of 0, 1s

	void boolArray(int n, int * odata, const int * idata){

		for (int i{0}; i < n; ++i){
			odata[i] = (idata[i] == 0) ? 0 : 1;
		}
	}
        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
	// implement the simplest for loop to do the sum of prior elements
        void scanNoTimer(int n, int *odata, const int *idata) {
		 int prior {0};
		 for (int i{0}; i < n; ++i)
		 {
			 *(odata+i) = prior;
			 prior = prior + idata[i];
		 }
        }
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
		    scanNoTimer(n, odata, idata);
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
                int size {0};
		for (int i {0}; i < n; ++i) 
		{
			if ( idata[i] != 0)  {
				odata[size++] = idata[i];
			}
		}

  	        timer().endCpuTimer();
            return size;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        
	//	timer().startCpuTimer();
		// allocation is costly and may dominate the time.
		int * boolarray { new int[n]};
		timer().startCpuTimer();
		// turn idata into an array of 0 and 1s
                boolArray(n, boolarray, idata);
		// odata is now the exclusive prefix sum.
		scanNoTimer(n, odata, boolarray);
		int lastIndex{-1};
		for (int i {0}; i < n; ++i)
		{
                      if ( boolarray[i] ) {
			         lastIndex = odata[i];
		          odata[lastIndex] = idata[i];
		      }
		}
	        timer().endCpuTimer();
                return lastIndex + 1;
        }
    }
}
