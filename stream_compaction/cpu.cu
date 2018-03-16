#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        //using StreamCompaction::Common::PerformanceTimer;
 /*       PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }*/

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        
		void scan(int n, int *odata, const int *idata) {
	        
			//timer().startCpuTimer();
            // TODO
			// Exclusive scan, first element is 0
			scanExclusivePrefixSum(n, odata, idata);
	        //timer().endCpuTimer();
        
		}

		// Scan implementation to avoid the CPU timer error
		void scanExclusivePrefixSum(int n, int *odata, const int *idata) {
		
			odata[0] = 0;
			for (int i = 1; i < n; ++i) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
	
		}

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        
		int compactWithoutScan(int n, int *odata, const int *idata) {
	    
			//timer().startCpuTimer();
            // TODO
			int count = 0;
			for (int i = 0; i < n; ++i) {
				if (idata[i] != 0) {
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
	    
			//timer().startCpuTimer();
	        // TODO
			int count = 0;
			int *mappedArray = new int[n];
			int *scannedArray = new int[n];

			// Map the data
			map(n, mappedArray, idata);

			// Scan the mapped data
			scanExclusivePrefixSum(n, scannedArray, mappedArray);

			// Scatter
			count = scatter(n, mappedArray, scannedArray, idata, odata);
	        
			//timer().endCpuTimer();
            return count;
 
		}

		/**
		* CPU stream compaction part: Mapping
		*
		* @based on the defined rules (here we map non zero values) assigns 1 or 0 for a given element in the idata array.
		*/
		
		void map(int n, int *mapped, const int *idata) {
		
			for (int i = 0; i < n; ++i) {
				if (idata[i] != 0) {
					mapped[i] = 1;
				}
				else {
					mapped[i] = 0;
				}
			}
		
		}

		/**
		* CPU stream compaction part: Scatter
		*
		* @scatters the input elements to the output vector using the addresses generated by the scan.
		*  returns the number of elements remaining after the scan. 
		*/
		
		int scatter(int n, int *mapped, int *scanned, const int *idata, int *odata) {
		
			int count = 0;
			for (int i = 0; i < n; ++i) {
				if (mapped[i] == 1) {
					odata[scanned[i]] = idata[i];
					count++;
				}
			}
			return count;
		}
    
	}

}
