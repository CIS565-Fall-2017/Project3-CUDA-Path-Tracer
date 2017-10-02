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

#define blockSize 32
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define USE_CUDA_DEV_SYNC 0

        __global__ void shiftRight(int n, int *odata, const int *idata) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;; // TODO?
          if (idx < n) {
            odata[idx] = (idx == 0) ? 0 : idata[idx - 1];
          }
        }

        __global__ void scanIteration(int n, int *odata, const int *idata, const int offset) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;; // TODO?
          if (idx < n) {
            odata[idx] = idata[idx] + ((idx >= offset) ? idata[idx - offset] : 0);
          }
        }

        int *dev_bufA;
        int *dev_bufB;

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          cudaMalloc((void**)&dev_bufA, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_bufA error!!!");
          cudaMalloc((void**)&dev_bufB, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_bufB error!!!");
          cudaMemcpy(dev_bufB, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAErrorWithLine("memcpy dev_bufB error!!!");
          dim3 numBlocks((n + blockSize - 1) / blockSize);
          
          timer().startGpuTimer();
          // TODO
          // shift right by one, set [0] to 0
          shiftRight<<<numBlocks, blockSize>>>(n, dev_bufA, dev_bufB);
#if USE_CUDA_DEV_SYNC
          cudaDeviceSynchronize();
          checkCUDAErrorWithLine("sync error");
#endif

          // start scan iterations
          int pingPongCount = 0;
          for (int offset = 1; offset < n; offset *= 2) {
            pingPongCount = 1 - pingPongCount;
            if (pingPongCount) {
              scanIteration << <numBlocks, blockSize >> >(n, dev_bufB, dev_bufA, offset);
            }
            else {
              scanIteration << <numBlocks, blockSize >> >(n, dev_bufA, dev_bufB, offset);
            }
#if USE_CUDA_DEV_SYNC
            cudaDeviceSynchronize();
            checkCUDAErrorWithLine("sync error");
#endif
          }
          timer().endGpuTimer();

          if (pingPongCount) {
            // output is in bufB
            cudaMemcpy(odata, dev_bufB, n * sizeof(int), cudaMemcpyDeviceToHost);
          }
          else {
            // output is in bufA
            cudaMemcpy(odata, dev_bufA, n * sizeof(int), cudaMemcpyDeviceToHost);
          }
          cudaFree(dev_bufA);
          cudaFree(dev_bufB);
        }
    }
}
