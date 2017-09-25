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
			thrust::device_vector<int> dev_idata(idata, idata + n);
			thrust::device_vector<int> dev_odata(odata, odata + n);
            timer().startGpuTimer();
			thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();
			thrust::copy(dev_odata.begin(), dev_odata.end(), odata);
        }
    }
}
