#include <cstdio>
#include "cpu.h"

#include "common.h"


/*
 * This stream compaction method will remove 0s from an array of ints.
*/
namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum). Compute an EXCLUSIVE prefix sum
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
			// TODO

			bool timerHasStartedElsewhere = false;
			try
			{
				//If this fails (b/c it was called elsewhere), it'll go into the catch
				timer().startCpuTimer();	
			}
			catch (std::runtime_error &e)
			{
				timerHasStartedElsewhere = true;
			}

			odata[0] = 0;
			for (int k = 1; k < n; k++)
			{
				odata[k] = odata[k - 1] + idata[k - 1];
			}

			//Only want to call endTimer if startTimer hasn't been called elsewhere
			if (!timerHasStartedElsewhere)
			{
				timer().endCpuTimer();
			}

        }//end scan function

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.

		 * Notes:
				Compute temp array containing 0's and 1's whether element meets criteria

         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            
			// TODO

			//Fill odata with elements from idata that aren't 0 (if any, this array's size will be less than idata)
			int counter = 0;

			for (int k = 0; k < n; k++)
			{
				if (idata[k] != 0)
				{
					odata[counter] = idata[k];
					counter++;
				}
			}

			timer().endCpuTimer();

			return counter;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.

		 * Notes:
				Map the input array to an array of 0s and 1s, scan it, 
				and use scatter to produce the output. 
				You will need a CPU scatter implementation for this 
				(see slides or GPU Gems chapter for an explanation).
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        
			// TODO
			//Map input array to temp array of 0's and 1's
			//int* itemp = new int[n];	
			int* itemp = (int *)malloc(sizeof(int) * n); 
			int counter = 0;

			for (int k = 0; k < n; k++)
			{
				if (idata[k] != 0)
				{
					itemp[k] = 1;
					counter++;
				}
				else
				{
					itemp[k] = 0;
				}
			}

			//Scan temp array
			//int* otemp = new int[n];	
			int* otemp = (int *)malloc(sizeof(int) * n); 
			scan(n, otemp, itemp);


			//Use scatter to produce output
			//Scan array values act as destination indices in odata
			int counter2 = 0;
			for (int k = 0; k < n; k++)
			{
				if (itemp[k] == 1)
				{
					int otemp_index = otemp[k];
					int idata_value = idata[k];

					odata[otemp_index] = idata_value;
					counter2++;
				}
			}

	        timer().endCpuTimer();

			//delete[] itemp;
			//delete[] otemp;
			free(itemp);
			free(otemp);

            return counter;
        }
    }
}
