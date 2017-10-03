#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
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
			thrust::host_vector<int> thrust_idata(n);
			thrust::host_vector<int> thrust_odata(n);
			thrust::copy(idata, idata + n, thrust_idata.begin());
			thrust::device_vector<int> thrust_dev_idata = thrust_idata;
			thrust::device_vector<int> thrust_dev_odata(n);

            timer().startGpuTimer();

			thrust::exclusive_scan(thrust_dev_idata.begin(), thrust_dev_idata.end(), thrust_dev_odata.begin());

            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();

			thrust::copy(thrust_dev_odata.begin(), thrust_dev_odata.end(), odata);
        }
    }
}
