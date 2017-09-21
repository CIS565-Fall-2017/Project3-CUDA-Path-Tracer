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
			thrust::device_vector<int> dv_in(idata, idata + n);
			thrust::device_vector<int> dv_out(odata, odata + n);
			thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            timer().endGpuTimer();

			thrust::copy(dv_out.begin(), dv_out.end(), odata);
        }
    }
}
