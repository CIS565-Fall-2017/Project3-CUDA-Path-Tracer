#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radixSort.h"
#include "efficient.h"

namespace StreamCompaction {
  namespace RadixSort {
    // access to "efficient" scan
    using StreamCompaction::Efficient::scan;
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }

#define blockSize 32
#define MAX_BLOCK_SIZE 32
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define USE_CUDA_DEV_SYNC 0
#define BITS_IN_INT (sizeof(int) * 8)
#define INT_WITH_HIGHEST_BIT_SET (1 << (BITS_IN_INT - 1))

    /* Sorts a tile of size _tileSize_ using radix sort in ascending order 
     * if _ascending_ is true, descending otherwise. 
     * This kernel should be called in blocks with size == tileSize, due to
     * the use of __syncthreads(). */
    /* could do the whole thing at once, instead of tiles then merge?? */
    __global__ void radixSortTile(int tileSize, int *odata, const int *idata, bool ascending) {

    }

    __global__ void computeEnumerate(int n, int *enumBuf, const int *originalData, const int passMask) {
      int idx = threadIdx.x + (blockDim.x * blockIdx.x);
      if (idx < n) {
        // idata[idx] & passMask is bit, so "negate" it
        enumBuf[idx] = (originalData[idx] & passMask) == 0 ? 1 : 0;
      }
    }

    __global__ void computeTotalFalses(int n, int *totalFalsesBuf, const int *enumBuf, const int *falseBuf) {
      *totalFalsesBuf = enumBuf[n - 1] + falseBuf[n - 1];
    }

    __global__ void computeTrueAndScatter(int n, int *odata, const int *falseBuf, const int *originalData, const int *totalFalsesBuf, const int passMask, bool ascending) {
      int idx = threadIdx.x + (blockDim.x * blockIdx.x);
      if (idx < n) {
        int originalValue = originalData[idx];
        // if masked bit is 0 and sorting by ascending, or if bit is 1 and sorting by descending
        if (((originalValue & passMask) == 0) == ascending) {
          // write to false idx
          odata[falseBuf[idx]] = originalValue;
        }
        else {
          // write to true idx
          odata[idx - falseBuf[idx] + *totalFalsesBuf] = originalValue;
        }
      }
    }

    __global__ void findFirstHighBit(int n, int *highBits, const int *idata) {
      int idx = threadIdx.x + (blockDim.x * blockIdx.x);
      if (idx < n) {
        int val = idata[idx];
        if (val == 0) {
          return;
        }
        int mask = INT_WITH_HIGHEST_BIT_SET;
        for (int i = BITS_IN_INT - 1; i >= 0; --i) {
          if (val & mask) {
            highBits[i] = 1;
            return;
          }
          mask >>= 1;
        }
      }
    }

    __global__ void  findLargestHighBit(int *result, const int *highBits) {
      for (int i = BITS_IN_INT - 1; i >= 0; --i) {
        if (highBits[i]) {
          *result = i;
          return;
        }
      }
    }

    void radixSort(int n, int *odata, const int *idata, bool ascending) {
      int *dev_highBitsBuf; 
      int *dev_originalData;
      int *dev_enumBuf;
      int *dev_falseBuf;
      int *dev_totalFalsesBuf;
      int *dev_postScatterBuf;      

      cudaMalloc((void**)&dev_highBitsBuf, BITS_IN_INT * sizeof(int));
      checkCUDAErrorWithLine("malloc dev_highBitsBuf error!!!");

      cudaMemset(dev_highBitsBuf, 0, BITS_IN_INT * sizeof(int));
      checkCUDAErrorWithLine("memset dev_highBitsBuf to 0 error!!!");

      cudaMalloc((void**)&dev_originalData, n * sizeof(int));
      checkCUDAErrorWithLine("malloc dev_originalData error!!!");

      cudaMalloc((void**)&dev_enumBuf, n * sizeof(int));
      checkCUDAErrorWithLine("malloc dev_enumBuf error!!!");

      // needs to be power of 2 for scan to work
      const int falseBufSize = 1 << ilog2ceil(n);

      cudaMalloc((void**)&dev_falseBuf, falseBufSize * sizeof(int));
      checkCUDAErrorWithLine("malloc dev_falseBuf error!!!");

      if (falseBufSize != n) {
        cudaMemset(dev_falseBuf, 0, falseBufSize * sizeof(int));
        checkCUDAErrorWithLine("memset dev_falseBuf to 0 error!!!");
      }

      cudaMalloc((void**)&dev_totalFalsesBuf, sizeof(int));
      checkCUDAErrorWithLine("malloc dev_totalFalsesBuf error!!!");

      cudaMalloc((void**)&dev_postScatterBuf, n * sizeof(int));
      checkCUDAErrorWithLine("malloc dev_postScatterBuf error!!!");

      cudaMemcpy(dev_originalData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAErrorWithLine("memcpy dev_originalData from host error!!!");

      dim3 numBlocks((n + blockSize - 1) / blockSize);

      int bitIdx;
      int passMask = 1;
      int maxBitIdx;

      timer().startGpuTimer();

      // find how many bits to search
      findFirstHighBit << <numBlocks, blockSize >> > (n, dev_highBitsBuf, dev_originalData);
      // store result in dev_highBitsBuf itself -- no race conditions
      findLargestHighBit<<<dim3(1), 1>>>(dev_highBitsBuf, dev_highBitsBuf);

      cudaMemcpy(&maxBitIdx, dev_highBitsBuf, sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAErrorWithLine("memcpy maxBitIdx error!!!");

      for (bitIdx = 0; bitIdx <= maxBitIdx; ++bitIdx) {
        // compute enumerate
        computeEnumerate << <numBlocks, blockSize >> >(n, dev_enumBuf, dev_originalData, passMask);
        // scan
        cudaMemcpy(dev_falseBuf, dev_enumBuf, n * sizeof(int), cudaMemcpyDeviceToDevice);
        checkCUDAErrorWithLine("memcpy enum to false error!!!");

        scan(n, dev_falseBuf, dev_enumBuf, true);
        checkCUDAErrorWithLine("scan error!!!");
        // compute totalFalses
        computeTotalFalses<<<dim3(1), 1>>>(n, dev_totalFalsesBuf, dev_enumBuf, dev_falseBuf);
        // compute trues and scatter
        computeTrueAndScatter << <numBlocks, blockSize >> > (n, dev_postScatterBuf, dev_falseBuf, dev_originalData, dev_totalFalsesBuf, passMask, ascending);
        
        if (bitIdx < maxBitIdx) {
          // if not last iteration
          cudaMemcpy(dev_originalData, dev_postScatterBuf, n * sizeof(int), cudaMemcpyDeviceToDevice);
          checkCUDAErrorWithLine("memcpy postScatter to originalData error!!!");

          // move passMask to next bit
          passMask <<= 1;
        }
      }

      timer().endGpuTimer();

      cudaMemcpy(odata, dev_postScatterBuf, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAErrorWithLine("memcpy dev_postScatterBuf to host error!!!");

      cudaFree(dev_highBitsBuf);
      checkCUDAErrorWithLine("free dev_highBitsBuf error!!!");

      cudaFree(dev_originalData);
      checkCUDAErrorWithLine("free dev_originalData error!!!");

      cudaFree(dev_enumBuf);
      checkCUDAErrorWithLine("free dev_enumBuf error!!!");

      cudaFree(dev_falseBuf);
      checkCUDAErrorWithLine("free dev_falseBuf error!!!");

      cudaFree(dev_totalFalsesBuf);
      checkCUDAErrorWithLine("free dev_scatteredSize error!!!");

      cudaFree(dev_postScatterBuf);
      checkCUDAErrorWithLine("free dev_postScatterBuf error!!!");

    }

  }
}
