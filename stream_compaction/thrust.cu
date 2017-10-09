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
		// read about cuda host/device vectors.  This way uses host and device
                // vector iterators.
        void scan(int n, int *odata, const int *idata) {
		  thrust::host_vector<int> myvec(n);
		  thrust::copy(idata, idata + n, myvec.begin());
		  thrust::device_vector<int> dev_idata = myvec;
          timer().startGpuTimer();
		  thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), 
			                    dev_idata.begin());
		  timer().endGpuTimer();
		  myvec = dev_idata;
		  thrust::copy(myvec.begin(), myvec.end(), odata);
        }
    }
}
