#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace EfficientShared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#define blockSize 32
#define MAX_BLOCK_SIZE 32
#define MAX_CHUNK_SIZE 32
#define checkCUDAErrorWithLine(msg) ((void)0) 
        //checkCUDAError(msg, __LINE__)
#define USE_CUDA_DEV_SYNC 0

/* Macros below adapated from: 
 * https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html/
 * Example 39-3 */
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

        __global__ void scanChunk(int n, int *g_odata, const int *g_idata) {
          extern __shared__ float sharedBuf[];
          int idx = threadIdx.x;
          int offset = 1;
          // compute indices and offsets to write to banks without conflict
          int aIdx = idx;
          int bIdx = idx + n / 2;
          int bankOffsetA = CONFLICT_FREE_OFFSET(aIdx);
          int bankOffsetB = CONFLICT_FREE_OFFSET(bIdx);
          sharedBuf[aIdx + bankOffsetA] = g_idata[aIdx];
          sharedBuf[bIdx + bankOffsetB] = g_idata[bIdx];
          // begin up-sweep
          for (int d = n / 2; d > 0; d /= 2) {
            __syncthreads();
            if (idx < d) {
              aIdx = offset * (2 * idx + 1) - 1;
              bIdx = aIdx + offset;//offset * (2 * idx + 2) - 1;
              aIdx += CONFLICT_FREE_OFFSET(aIdx);
              bIdx += CONFLICT_FREE_OFFSET(bIdx);

              sharedBuf[bIdx] += sharedBuf[aIdx];
            }

            offset *= 2;
          }

          // set last idx to 0
          if (idx == 0) { 
            sharedBuf[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; 
          }

          for (int d = 1; d < n; d *= 2) {
            offset /= 2;
            __syncthreads();
            if (idx < d) {
              aIdx = offset * (2 * idx + 1) - 1;
              bIdx = aIdx + offset;//offset * (2 * idx + 2) - 1;
              aIdx += CONFLICT_FREE_OFFSET(aIdx);
              bIdx += CONFLICT_FREE_OFFSET(bIdx);

              float originalNodeValue = sharedBuf[aIdx];
              sharedBuf[aIdx] = sharedBuf[bIdx];
              sharedBuf[bIdx] += originalNodeValue;
            }
          }

          __syncthreads();

          g_odata[aIdx] = sharedBuf[aIdx + bankOffsetA];
          g_odata[bIdx] = sharedBuf[bIdx + bankOffsetB];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         * internalUse specifies whether this is used as a helper function,
         * for example, in compact. If so, it assumes idata and odata are in
         * device memory and does not use gpuTimer.
         */
        void scan(int n, int *odata, const int *idata) {
          if (n == 1) {
            odata[0] = 0;
            return;
          }
          // TODO: handle n <= 2 ???
          // nearest power of two
          const int bufSize = 1 << ilog2ceil(n);
          
          int *dev_buf;

          cudaMalloc((void**)&dev_buf, bufSize * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_buf error!!!");

          if (n < bufSize) {
            cudaMemset(dev_buf + n, 0, (bufSize - n) * sizeof(int));
            checkCUDAErrorWithLine("memset dev_buf to 0 error!!!");
          }

          cudaMemcpy(dev_buf, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAErrorWithLine("memcpy dev_buf error!!!");

          timer().startGpuTimer();

          scanChunk<<<dim3(1), bufSize / 2, bufSize * sizeof(int)>>>(bufSize, dev_buf, dev_buf);
          checkCUDAErrorWithLine("scan chunk error!!!");

          timer().endGpuTimer();


          cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);
          checkCUDAErrorWithLine("memcpy dev_buf to host error!!!");
          cudaFree(dev_buf);
          checkCUDAErrorWithLine("free dev_buf error!!!");
          
      
        }

    }
}
