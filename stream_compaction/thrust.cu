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
      // TODO use `thrust::exclusive_scan`
      // example: for device_vectors dv_in and dv_out:
      // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

      int *idataSwap;
      int *odataSwap;

      cudaMalloc((void**)&idataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for idataSwap failed!");

      cudaMalloc((void**)&odataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for odataSwap failed!");

      // Copy from CPU to GPU
      cudaMemcpy(idataSwap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy failed!");

      timer().startGpuTimer();

      thrust::device_ptr<int> dev_thrust_idata(idataSwap);
      thrust::device_ptr<int> dev_thrust_odata(odataSwap);

      thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);

      timer().endGpuTimer();

      // Copy from GPU back to CPU
      cudaMemcpy(odata, odataSwap, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(idataSwap);
      cudaFree(odataSwap);
    }
  }
}
