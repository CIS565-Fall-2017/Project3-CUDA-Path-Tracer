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
			int* dv_in, *dv_out;
			cudaMalloc((void**)&dv_in, sizeof(int) * n);
			cudaMalloc((void**)&dv_out, sizeof(int) * n);

			cudaMemcpy(dv_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			thrust::device_ptr<int> dv_in_thrust(dv_in);
			thrust::device_ptr<int> dv_out_thrust(dv_out);
			thrust::exclusive_scan(dv_in_thrust, dv_in_thrust + n, dv_out_thrust);

            timer().startGpuTimer();
			thrust::exclusive_scan(dv_in_thrust, dv_in_thrust + n, dv_out_thrust);
            timer().endGpuTimer();

			cudaMemcpy(odata, dv_out, sizeof(int) * n, cudaMemcpyDeviceToHost);

			cudaFree(dv_in);
			cudaFree(dv_out);
        }
    }
}
