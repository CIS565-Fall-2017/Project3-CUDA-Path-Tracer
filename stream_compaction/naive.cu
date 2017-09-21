#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
  namespace Naive {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }

    __global__ void kernScan(int n, int* odata, int* idata, int shift) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      if (index >= shift) {
        odata[index] = idata[index] + idata[index - shift];
      }
      else {
        odata[index] = idata[index];
      }
    }

    __global__ void kernExclusiveShift(int n, int* odata, int* idata) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      if (index > 0) {
        odata[index] = idata[index - 1];
      }
      else {
        odata[index] = 0;
      }
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int *odata, const int *idata) {
            
      dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

      int* odataSwap;
      cudaMalloc((void**)&odataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for odataSwap failed");

      int* idataSwap;
      cudaMalloc((void**)&idataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for idataSwap failed");

      // Copy from CPU to GPU
      cudaMemcpy(odataSwap, odata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for odataSwap failed");

      cudaMemcpy(idataSwap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for idataSwap failed");

      timer().startGpuTimer();

      for (int depth = 1; depth <= ilog2ceil(n); depth++) {
        int shift = 1;
        if (depth > 1) {
          shift = 2 << (depth - 2);
        }

        kernScan << <fullBlocksPerGrid, blockSize >> >(n, odataSwap, idataSwap, shift);
        checkCUDAError("kernScan failed");

        // Swap buffers for next iteration
        cudaMemcpy(idataSwap, odataSwap, n * sizeof(int), cudaMemcpyDeviceToDevice);
        checkCUDAError("cudaMemcpy to swap buffers failed");
      }

      kernExclusiveShift << <fullBlocksPerGrid, blockSize >> >(n, odataSwap, idataSwap);
      checkCUDAError("kernExclusiveShift failed");

      // Copy from GPU back to CPU
      cudaMemcpy(odata, odataSwap, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy for odataSwap failed");

      timer().endGpuTimer();

      cudaFree(odataSwap);
      cudaFree(idataSwap);
    }
  }
}
