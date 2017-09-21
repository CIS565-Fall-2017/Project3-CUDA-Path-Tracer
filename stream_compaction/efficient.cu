#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
  namespace Efficient {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }

    __global__ void kernUpSweep(int n, int* idata, int shift) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      int offset = (shift << 1);
      if (index % offset == 0 && index + offset <= n) {
        idata[index + offset - 1] += idata[index + shift - 1];
      }
    }

    __global__ void kernDownSweep(int n, int* idata, int shift) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      int offset = (shift << 1);
      if (index % offset == 0 && index + offset <= n) {
        int temp = idata[index + shift - 1];
        idata[index + shift - 1] = idata[index + offset - 1];
        idata[index + offset - 1] += temp;
      }
    }

    /**
      * Performs prefix-sum (aka scan) on idata, storing the result into odata.
      */
    void scan(int n, int *odata, const int *idata) {
      int maxN = (1 << ilog2ceil(n));
      dim3 fullBlocksPerGrid((maxN + blockSize - 1) / blockSize);

      int* idataSwap;

      cudaMalloc((void**)&idataSwap, maxN * sizeof(int));
      checkCUDAError("cudaMalloc for idataSwap failed");

      cudaMemset(idataSwap, 0, maxN * sizeof(int));
      checkCUDAError("cudaMemset for idataSwap failed");

      // Copy from CPU to GPU
      cudaMemcpy(idataSwap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for idataSwap failed");

      timer().startGpuTimer();

      // Up-sweep
      for (int depth = 0; depth < ilog2ceil(n); depth++) {
        int shift = (1 << depth);

        kernUpSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idataSwap, shift);
        checkCUDAError("kernUpSweep failed");
      }

      cudaMemset(idataSwap + maxN - 1, 0, sizeof(int));
        
      // Down-sweep
      for (int depth = ilog2ceil(n) - 1; depth >= 0; depth--) {
        int shift = (1 << depth);

        kernDownSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idataSwap, shift);
        checkCUDAError("kernUpSweep failed");
      }

      timer().endGpuTimer();

      // Copy from GPU back to CPU
      cudaMemcpy(odata, idataSwap, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy for idataSwap failed");

      cudaFree(idataSwap);
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
        
      dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

      // Allocate extra buffers
      int* odataSwap;
      cudaMalloc((void**)&odataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for odataSwap failed");

      int* idataSwap;
      cudaMalloc((void**)&idataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for idataSwap failed");

      int* boolsArr;
      cudaMalloc((void**)&boolsArr, n * sizeof(int));
      checkCUDAError("cudaMalloc for boolsArr failed");

      int* indicesArr;
      cudaMalloc((void**)&indicesArr, n * sizeof(int));
      checkCUDAError("cudaMalloc for scan_result failed");

      // Copy from CPU to GPU
      cudaMemcpy(odataSwap, odata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for odataSwap failed");

      cudaMemcpy(idataSwap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for idataSwap failed");

      timer().startGpuTimer();

      // Map input array to a temp array of 0s and 1s
      StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, boolsArr, idataSwap);
      checkCUDAError("kernMapToBoolean failed");

      // Scan
      scan(n, indicesArr, boolsArr);

      // Scatter
      StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, odataSwap, idataSwap, boolsArr, indicesArr);
      checkCUDAError("kernScatter failed");

      timer().endGpuTimer();

      // Copy over compacted data from GPU to CPU
      cudaMemcpy(odata, odataSwap, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy for odataSwap failed");

		  // Grab remaining number of elements
		  int remainingNBools = 0;
		  cudaMemcpy(&remainingNBools, boolsArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

		  int remainingNIndices = 0;
		  cudaMemcpy(&remainingNIndices, indicesArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	
      cudaFree(odataSwap);
      cudaFree(idataSwap);
      cudaFree(boolsArr);
      cudaFree(indicesArr);
        
      return remainingNBools + remainingNIndices;
    }

    __global__ void kernMapToBooleanPaths(int n, int *bools, const PathSegment *idata) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      if (idata[index].remainingBounces != 0) {
        bools[index] = 1;
      }
      else {
        bools[index] = 0;
      }
    }

    __global__ void kernScatterPaths(int n, PathSegment *odata,
      const PathSegment *idata, const int *bools, const int *indices) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      if (bools[index] == 1) {
        odata[indices[index]] = idata[index];
      }
    }

    void scanPaths(int n, int *odata, const int *idata) {
      int maxN = (1 << ilog2ceil(n));
      dim3 fullBlocksPerGrid((maxN + blockSize - 1) / blockSize);

      int* idataSwap;

      cudaMalloc((void**)&idataSwap, maxN * sizeof(int));
      checkCUDAError("cudaMalloc for idataSwap failed");

      cudaMemset(idataSwap, 0, maxN * sizeof(int));
      checkCUDAError("cudaMemset for idataSwap failed");

      // Copy from GPU to GPU
      cudaMemcpy(idataSwap, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
      checkCUDAError("cudaMemcpy for idataSwap failed");


      // Up-sweep
      for (int depth = 0; depth < ilog2ceil(n); depth++) {
        int shift = (1 << depth);

        kernUpSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idataSwap, shift);
        checkCUDAError("kernUpSweep failed");
      }

      cudaMemset(idataSwap + maxN - 1, 0, sizeof(int));

      // Down-sweep
      for (int depth = ilog2ceil(n) - 1; depth >= 0; depth--) {
        int shift = (1 << depth);

        kernDownSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idataSwap, shift);
        checkCUDAError("kernUpSweep failed");
      }


      // Copy from GPU back to GPU
      cudaMemcpy(odata, idataSwap, n * sizeof(int), cudaMemcpyDeviceToDevice);
      checkCUDAError("cudaMemcpy for idataSwap failed");

      cudaFree(idataSwap);
    }

    /**
    * Performs stream compaction on paths, storing the result into odata.
    * All zeroes are discarded.
    *
    * @param n      The number of elements in idata.
    * @param odata  The array into which to store elements.
    * @param idata  The array of elements to compact.
    * @returns      The number of elements remaining after compaction.
    */
    int compactPaths(int n, PathSegment *odata, PathSegment *idata, int* bools_arr, int* indices_arr) {

      dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

      // Copy from CPU to CPU
      cudaMemcpy(odata, idata, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
      checkCUDAError("cudaMemcpy for odata failed");


      // Map input array to a temp array of 0s and 1s
      kernMapToBooleanPaths << <fullBlocksPerGrid, blockSize >> >(n, bools_arr, idata);
      checkCUDAError("kernMapToBoolean failed");

      // Scan
      scan(n, indices_arr, bools_arr);

      // Scatter
      kernScatterPaths << <fullBlocksPerGrid, blockSize >> >(n, odata, idata, bools_arr, indices_arr);
      checkCUDAError("kernScatter failed");

      // Copy from CPU to CPU
      cudaMemcpy(idata, odata, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
      checkCUDAError("cudaMemcpy for odataSwap failed");

      // Grab remaining number of elements
      int remainingNBools = 0;
      cudaMemcpy(&remainingNBools, bools_arr + n - 1, sizeof(PathSegment), cudaMemcpyDeviceToHost);

      int remainingNIndices = 0;
      cudaMemcpy(&remainingNIndices, indices_arr + n - 1, sizeof(PathSegment), cudaMemcpyDeviceToHost);

      return remainingNBools + remainingNIndices;
    }
  }
}
