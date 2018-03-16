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
         */

		void scan(int n, int *odata, const int *idata) {

			int size = n * sizeof(int);

			// Wrap device buffers into thrust pointers
			thrust::device_vector<int> iThrustVector(idata, idata + n);
			thrust::device_vector<int> oThrustVector(odata, odata + n);

			timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            thrust::exclusive_scan(iThrustVector.begin(), iThrustVector.end(), oThrustVector.begin());
            timer().endGpuTimer();
		
			thrust::copy(oThrustVector.begin(), oThrustVector.end(), odata);
		}
   
	}

}
