#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.

		 Notes:
				This should be a very short function which wraps a call to the 
				Thrust library function thrust::exclusive_scan(first, last, result).

				To measure timing, be sure to exclude memory operations by passing 
				exclusive_scan a thrust::device_vector (which is already allocated on the GPU). 
				You can create a thrust::device_vector by creating a 
				thrust::host_vector from the given pointer, then casting it.

				For thrust stream compaction, take a look at thrust::remove_if. 
				It's not required to analyze thrust::remove_if but you're encouraged to do so.

				Thrust Documentation: http://docs.nvidia.com/cuda/thrust/index.html
				Thrust Quick Start Guide: https://github.com/thrust/thrust/wiki/Quick-Start-Guide
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			//Create host vectors
			thrust::host_vector<int> host_vector_in(n);
																
			//Use thrust::copy to copy data between host and device
			thrust::copy(idata, idata + n, host_vector_in.begin());

			//Create device vectors to put into thrust::exclusive_scan.
			//Cast the host vector to device vector
			thrust::device_vector<int> dev_vector_in = host_vector_in;
			thrust::device_vector<int> dev_vector_out(n);// = host_vector_out;

			//Can also do it this way -- but isn't any faster
			//thrust::device_vector<int> dev_vector_in(idata, idata + n);
			//thrust::device_vector<int> dev_vector_out(odata, odata + n);

			//Placing an exclusive_scan call once here outside the timer and once inside the timer block 
			//apparently neutralizes the difference between POT and NPOT sized arrays 

			timer().startGpuTimer();

			thrust::exclusive_scan(dev_vector_in.begin(), dev_vector_in.end(), dev_vector_out.begin());

			timer().endGpuTimer();

			//Copy the data back to odata
			thrust::copy(dev_vector_out.begin(), dev_vector_out.end(), odata);
        }
    }
}
