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
			// Do not time memory allocation
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			int* dev_idata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			thrust::device_ptr<int> dv_out(dev_odata);
			thrust::device_ptr<int> dv_in(dev_idata);
            
			// Time everything else
			timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			thrust::exclusive_scan(dv_in, dv_in + n, dv_out);
            timer().endGpuTimer();

			// Get the return value off of the device and free memory.
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
        }
    }
}
