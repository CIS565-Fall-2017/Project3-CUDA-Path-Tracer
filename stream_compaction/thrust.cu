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
          int *dev_idata;

          cudaMalloc((void **)&dev_idata, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_idata!!!");

          cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAErrorWithLine("memcpy dev_idata from host!!!");

          thrust::device_ptr<int> dev_thrust_idata(dev_idata);

          // pass in cpu pointers here

          thrust::device_vector<int> dev_vec_idata(dev_idata, dev_idata + n);

          timer().startGpuTimer();
          thrust::exclusive_scan(dev_vec_idata.begin(), dev_vec_idata.end(), dev_vec_idata.begin());
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
          timer().endGpuTimer();

          thrust::copy(dev_vec_idata.begin(), dev_vec_idata.end(), odata);

          cudaFree(dev_idata);
          checkCUDAErrorWithLine("free dev_idata!!!");

        }
    }
}
