#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
  namespace Naive {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }
    // TODO: __global__
    __global__ void kernScan(int n, const int pow, int *odata, const int *idata) {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= n) return;

      odata[index] = (index >= pow) ? idata[index - pow] + idata[index] : idata[index];
    }

    __global__ void kernInToEx(int n, int *odata, const int *idata) {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= n) return;

      odata[index] = (index == 0) ? 0 : idata[index - 1];
    }

    /**
      * Performs prefix-sum (aka scan) on idata, storing the result into odata.
      */
    void scan(int n, int *odata, const int *idata) {
      // Create device arrays
      int *dev_odata, *dev_idata;
      int nsize = n * sizeof(int);

      cudaMalloc((void**)&dev_odata, nsize);
      checkCUDAError("cudaMalloc for dev_odata failed!");

      cudaMalloc((void**)&dev_idata, nsize);
      checkCUDAError("cudaMalloc for dev_idata failed!");

      // Copy device arrays to device
      cudaMemcpy(dev_odata, odata, nsize, cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for dev_odata failed!");

      cudaMemcpy(dev_idata, idata, nsize, cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for dev_idata failed!");

      // Compute block per grid and thread per block
      dim3 numBlocks((n + blockSize - 1) / blockSize);
      dim3 numThreads(blockSize);

      timer().startGpuTimer();
      // Naive Scan - Creates inclusive scan output
      int levels = ilog2ceil(n);
      for (int d = 1; d <= levels; d++) {
        int pow = 1 << (d - 1);
        kernScan <<<numBlocks, numThreads >>> (n, pow, dev_odata, dev_idata);
        checkCUDAError("kernScan failed for level " + levels);
        std::swap(dev_odata, dev_idata);
      }

      // Convert inclusive scan to exclusive
      kernInToEx <<<numBlocks, numThreads >>> (n, dev_odata, dev_idata);
      checkCUDAError("kernInToEx failed!");
      
      timer().endGpuTimer();

      // Copy device arrays back to host
      cudaMemcpy(odata, dev_odata, nsize, cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy (device to host) for dev_odata failed!");

      // Free memory
      cudaFree(dev_odata);
      cudaFree(dev_idata);
      checkCUDAError("cudaFree failed!");
    }
  }
}
