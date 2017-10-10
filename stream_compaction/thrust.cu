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
          thrust::device_vector<int> inDevice(idata, idata + n);
          thrust::device_vector<int> outDevice(odata, odata + n);

          timer().startGpuTimer();
          thrust::exclusive_scan(inDevice.begin(), inDevice.end(), outDevice.begin());
          timer().endGpuTimer();

          thrust::copy(outDevice.begin(), outDevice.end(), odata);
        }
    }
}
