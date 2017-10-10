#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Kernel to pad the new array with 0s
        __global__ void kernPadWithZeros(const int n, const int nPad, int *dev_data) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index >= nPad || index < n) return;

          dev_data[index] = 0;
        }

        // Up-Sweep Kernel
        __global__ void kernUpSweep(const int n, const int pow, const int pow1, int *dev_data) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index * pow1 >= n) return;

          int idx = (index + 1) * pow1 - 1;
          dev_data[idx] += dev_data[idx - pow];
        }

        // Down-Sweep Kernel
        __global__ void kernDownSweep(const int n, const int pow, const int pow1, int *dev_data) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index * pow1 >= n) return;

          int idx = (index + 1) * pow1 - 1;
          int t = dev_data[idx - pow];
          dev_data[idx - pow] = dev_data[idx];
          dev_data[idx] += t;
        }

        void scan_implementation(const int nl, const dim3 numBlocks, const dim3 numThreads,
          const int nPad, int*dev_data) {

          for (int d = 0; d < nl; d++) {
            int pow = 1 << (d);
            int pow1 = 1 << (d + 1);
            dim3 nB((nPad / pow1 + blockSize - 1) / blockSize);
            kernUpSweep <<<nB, numThreads>>> (nPad, pow, pow1, dev_data);
            checkCUDAError("kernUpSweep failed!");
          }

          //dev_data[nPad - 1] = 0; // set last element to 0 before downsweep.. 
          int zero = 0;
          cudaMemcpy(dev_data + nPad - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAError("cudaMemcpy failed!");

          for (int d = nl - 1; d >= 0; d--) {
            int pow = 1 << (d);
            int pow1 = 1 << (d + 1);
            dim3 nB((nPad / pow1 + blockSize - 1) / blockSize);
            kernDownSweep <<<nB, numThreads>>> (nPad, pow, pow1, dev_data);
            checkCUDAError("kernDownSweep failed!");
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          int nSize = n * sizeof(int);
          int nl = ilog2ceil(n);
          int nPad = 1 << nl;
          int nPadSize = nPad * sizeof(int);

          // Compute blocks per grid and threads per block
          dim3 numBlocks((nPad + blockSize - 1) / blockSize);
          dim3 numThreads(blockSize);

          int *dev_data;
          cudaMalloc((void**)&dev_data, nPadSize);
          checkCUDAError("cudaMalloc for dev_data failed!");

          // Copy device arrays to device
          cudaMemcpy(dev_data, idata, nSize, cudaMemcpyHostToDevice); // use a kernel to fill 0s for the remaining indices..
          checkCUDAError("cudaMemcpy for dev_data failed!");
          
          // Fill the padded part of dev_data with 0s..
          kernPadWithZeros <<<numBlocks, numThreads>>> (n, nPad, dev_data);

          timer().startGpuTimer();
          // Work Efficient Scan - Creates exclusive scan output

          scan_implementation(nl, numBlocks, numThreads, nPad, dev_data);

          timer().endGpuTimer();

          // Copy device arrays back to host
          cudaMemcpy(odata, dev_data, nSize, cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy (device to host) for odata failed!");

          // Free memory
          cudaFree(dev_data);
          checkCUDAError("cudaFree failed!");
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
          int nSize = n * sizeof(int);
          int nl = ilog2ceil(n);
          int nPad = 1 << nl;
          int nPadSize = nPad * sizeof(int);

          int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
          cudaMalloc((void**)&dev_idata, nSize);
          checkCUDAError("cudaMalloc for dev_idata failed!");

          cudaMalloc((void**)&dev_odata, nSize);
          checkCUDAError("cudaMalloc for dev_odata failed!");

          cudaMalloc((void**)&dev_bools, nSize);
          checkCUDAError("cudaMalloc for dev_bools failed!");

          cudaMalloc((void**)&dev_indices, nPadSize);
          checkCUDAError("cudaMalloc for dev_indices failed!");

          // Copy device arrays to device
          cudaMemcpy(dev_idata, idata, nSize, /*cudaMemcpyHostToDevice*/cudaMemcpyDeviceToDevice);
          checkCUDAError("cudaMemcpy for dev_data failed!");

          dim3 numBlocks((n + blockSize - 1) / blockSize);
          dim3 numBlocksPadded((nPad + blockSize - 1) / blockSize);
          dim3 numThreads(blockSize);

          timer().startGpuTimer();
          
          // Create bools array
          StreamCompaction::Common::kernMapToBoolean <<<numBlocks, numThreads>>> (n, dev_bools, dev_idata);
          checkCUDAError("cudaMemcpy for kernMapToBoolean failed!");

          // Copy bools array to indices array - device to device
          cudaMemcpy(dev_indices, dev_bools, nSize, cudaMemcpyDeviceToDevice);
          checkCUDAError("cudaMemcpy for dev_indices failed!");
          // Pad the extended array with 0s
          kernPadWithZeros <<<numBlocks, numThreads>>> (n, nPad, dev_indices);
          checkCUDAError("cudaMemcpy for kernPadWithZeros failed!");

          // Work Efficient Scan
          scan_implementation(nl, numBlocksPadded, numThreads, nPad, dev_indices);
          
          // Scatter
          StreamCompaction::Common::kernScatter <<<numBlocks, numThreads>>> (n, dev_odata, dev_idata, dev_bools, dev_indices);
          checkCUDAError("cudaMemcpy for kernScatter failed!");

          timer().endGpuTimer();

          int newSize, indEnd, boolEnd;
          cudaMemcpy(&indEnd, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(&boolEnd, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
          newSize = indEnd + boolEnd;
          //printf("%d",newSize);

          // Copy device arrays back to host
          cudaMemcpy(odata, dev_odata, nSize, /*cudaMemcpyDeviceToHost*/cudaMemcpyDeviceToDevice);
          checkCUDAError("cudaMemcpy (device to host) for odata failed!");

          // Free memory
          cudaFree(dev_idata);
          cudaFree(dev_odata);
          cudaFree(dev_bools);
          cudaFree(dev_indices);
          checkCUDAError("cudaFree failed!");
          return newSize;
        }
    }
}
