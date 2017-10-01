#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <src/sceneStructs.h>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#define blockSize 32
#define MAX_BLOCK_SIZE 16
#define checkCUDAErrorWithLine(msg) ((void)0) 
        //checkCUDAError(msg, __LINE__)
#define USE_CUDA_DEV_SYNC 0

        __global__ void upSweepIteration(int n, int *odata, const int offset, const int halfOffset) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
          int nodeIdx = (idx + 1) * offset - 1;
          if (nodeIdx < n) {
            odata[nodeIdx] = odata[nodeIdx] + odata[nodeIdx - halfOffset];
          }
        }

        __global__ void setRootToZero(int n, int *odata) {
          odata[n - 1] = 0;
        }

        __global__ void downSweepIteration(int n, int *odata, const int offset, const int halfOffset) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
          int nodeIdx = (idx + 1) * offset - 1;
          if (nodeIdx < n) {
            int originalNodeValue = odata[nodeIdx];
            odata[nodeIdx] = odata[nodeIdx] + odata[nodeIdx - halfOffset];
            odata[nodeIdx - halfOffset] = originalNodeValue;
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         * internalUse specifies whether this is used as a helper function,
         * for example, in compact. If so, it assumes idata and odata are in
         * device memory and does not use gpuTimer.
         */
        void scan(int n, int *odata, const int *idata, bool internalUse) {
          if (n == 1) {
            odata[0] = 0;
            return;
          }
          // TODO: handle n <= 2 ???
          // nearest power of two
          const int bufSize = 1 << ilog2ceil(n);
          
          int *dev_buf;
          if (internalUse) {
            dev_buf = odata;
          }
          else {
            cudaMalloc((void**)&dev_buf, bufSize * sizeof(int));
            checkCUDAErrorWithLine("malloc dev_buf error!!!");

            if (n != bufSize) {
              cudaMemset(dev_buf + n, 0, (bufSize - n) * sizeof(int));
              checkCUDAErrorWithLine("memset dev_buf to 0 error!!!");
            }

            cudaMemcpy(dev_buf, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("memcpy dev_buf error!!!");
          }



          int halfOffset = 1;
          int numThreads = bufSize / 2;
          dim3 numBlocks(1);
          int threadsPerBlock;

          cudaDeviceSynchronize();
          checkCUDAErrorWithLine("cuda sync error!!!");

          if (!internalUse) {
            timer().startGpuTimer();
          }
          // skip offset = n because we overwrite root's value anyway
          for (int offset = 2; offset < bufSize; offset *= 2) {
            if (numThreads > MAX_BLOCK_SIZE) {
              numBlocks.x = numThreads / MAX_BLOCK_SIZE;
              //numBlocks = dim3(numThreads / MAX_BLOCK_SIZE);
              threadsPerBlock = MAX_BLOCK_SIZE;
            }
            else {
              numBlocks.x = 1;
              //numBlocks = dim3(1);
              threadsPerBlock = numThreads;
            }
            upSweepIteration<<<numBlocks, threadsPerBlock>>>(bufSize, dev_buf, offset, halfOffset);
            checkCUDAErrorWithLine("upSweep error!!!");
            halfOffset = offset;
            numThreads /= 2;
          }

          setRootToZero << <dim3(1), 1 >> > (bufSize, dev_buf);

          int offset = bufSize;
          numThreads = 1;
          for (int halfOffset = bufSize / 2; halfOffset >= 1; halfOffset /= 2) {
            if (numThreads > MAX_BLOCK_SIZE) {
              numBlocks.x = numThreads / MAX_BLOCK_SIZE;
              //numBlocks = dim3(numThreads / MAX_BLOCK_SIZE);
              threadsPerBlock = MAX_BLOCK_SIZE;
            }
            else {
              numBlocks.x = 1;
              //numBlocks = dim3(1);
              threadsPerBlock = numThreads;
            }
            downSweepIteration << <numBlocks, threadsPerBlock >> >(bufSize, dev_buf, offset, halfOffset);
            checkCUDAErrorWithLine("downSweep error!!!");
            offset = halfOffset;
            numThreads *= 2;
          }

          if (!internalUse) {
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("memcpy dev_buf to host error!!!");
            cudaFree(dev_buf);
            checkCUDAErrorWithLine("free dev_buf error!!!");
          }
      
        }

        __global__ void map(int n, int *odata, const int *idata) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x; // TODO?
          if (idx < n) {
            odata[idx] = (idata[idx] != 0) ? 1 : 0;
          }
        }

        __global__ void scatter(int n, int *odata, const int *postMapData, const int *postScanData, const int *originalData) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x; // TODO?
          if (idx < n && postMapData[idx]) {
            odata[postScanData[idx]] = originalData[idx];
          }
        }

        __global__ void getCompactedSize(int n, int *odata, const int *postMapData, const int *postScanData) {
          *odata = postScanData[n - 1] + (postMapData[n - 1] ? 1 : 0);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
          int *dev_originalData;
          int *dev_postMapBuf;
          int *dev_postScanBuf;
          int *dev_postScatterBuf;
          int *dev_scatteredSize;

          cudaMalloc((void**)&dev_originalData, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_originalData error!!!");

          cudaMalloc((void**)&dev_postMapBuf, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_postMapBuf error!!!");

          // needs to be power of 2 for scan to work
          const int postScanBufSize = 1 << ilog2ceil(n);

          cudaMalloc((void**)&dev_postScanBuf, postScanBufSize * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_postScanBuf error!!!");

          if (postScanBufSize != n) {
            cudaMemset(dev_postScanBuf, 0, postScanBufSize * sizeof(int));
            checkCUDAErrorWithLine("memset dev_postScanBuf to 0 error!!!");
          }

          cudaMalloc((void**)&dev_postScatterBuf, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_postScatterBuf error!!!");

          cudaMalloc((void**)&dev_scatteredSize, sizeof(int));
          checkCUDAErrorWithLine("malloc dev_scatteredSize error!!!");

          cudaMemcpy(dev_originalData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAErrorWithLine("memcpy dev_originalData from host error!!!");

          dim3 numBlocks((n + blockSize - 1) / blockSize);

          timer().startGpuTimer();

          map<<<numBlocks, blockSize>>>(n, dev_postMapBuf, dev_originalData);
          checkCUDAErrorWithLine("map error!!!");

          cudaMemcpy(dev_postScanBuf, dev_postMapBuf, n * sizeof(int), cudaMemcpyDeviceToDevice);
          checkCUDAErrorWithLine("memcpy map to scan error!!!");

          scan(n, dev_postScanBuf, dev_postMapBuf, true);
          checkCUDAErrorWithLine("scan error!!!");
          scatter << <numBlocks, blockSize >> > (n, dev_postScatterBuf, dev_postMapBuf, dev_postScanBuf, dev_originalData);
          checkCUDAErrorWithLine("scatter error!!!");
          getCompactedSize<<<dim3(1), 1>>>(n, dev_scatteredSize, dev_postMapBuf, dev_postScanBuf);
          checkCUDAErrorWithLine("get size error!!!");
          
          timer().endGpuTimer();

          int scatteredSize;

          cudaMemcpy(&scatteredSize, dev_scatteredSize, sizeof(int), cudaMemcpyDeviceToHost);
          checkCUDAErrorWithLine("memcpy dev_scatteredSize to host error!!!");

          cudaMemcpy(odata, dev_postScatterBuf, scatteredSize * sizeof(int), cudaMemcpyDeviceToHost);
          checkCUDAErrorWithLine("memcpy dev_postScatterBuf to host error!!!");

          cudaFree(dev_originalData);
          checkCUDAErrorWithLine("free dev_originalData error!!!");

          cudaFree(dev_postMapBuf);
          checkCUDAErrorWithLine("free dev_postMapBuf error!!!");

          cudaFree(dev_postScanBuf);
          checkCUDAErrorWithLine("free dev_postScanBuf error!!!");

          cudaFree(dev_postScatterBuf);
          checkCUDAErrorWithLine("free dev_postScatterBuf error!!!");

          cudaFree(dev_scatteredSize);
          checkCUDAErrorWithLine("free dev_scatteredSize error!!!");

          return scatteredSize;
        }
    }
}
