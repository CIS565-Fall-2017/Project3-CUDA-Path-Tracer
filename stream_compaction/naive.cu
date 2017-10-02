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
		/*****************
		* Configuration *
		*****************/

		/*! Block size used for CUDA kernel launch. */
		#define blockSize 128

		// Implementation of a level's step in scan.
		// Find the location of the previous sum and add it to its new position.
		__global__ void kernScan(int numObjects, int* odata, int* idata, int level) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= numObjects) {
				return;
			}
			
			int previousSumIndex = powf(2, level);
			if (index < previousSumIndex) {
				odata[index] = idata[index];
			} else {
				odata[index] = idata[index] + idata[index - previousSumIndex];
			}
		}

		// Make this an exclusive prefix scan by making the first element 0.
		// Push all other elements back.
		__global__ void kernExclusivePrefix(int numObjects, int* odata, int* idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= numObjects) {
				return;
			}

			odata[0] = 0;
			if (index > 0) {
				odata[index] = idata[index - 1];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// Do not time memory allocation
			// "Create two device arrays. Swap them at each iteration: 
			// read from A and write to B, read from B and write to A."
			int* dev_A;
			cudaMalloc((void**)&dev_A, n * sizeof(int));
			// How to copy data to the GPU
			cudaMemcpy(dev_A, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			int* dev_B;
			cudaMalloc((void**)&dev_B, n * sizeof(int));
			// How to copy data to the GPU
			cudaMemcpy(dev_B, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			// Time everything else
            timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			// "You will have to do ilog2ceil(n) separate kernel invocations"
			int depthMax = ilog2ceil(n);
			for (int depth = 0; depth < depthMax; ++depth) {
				kernScan<<<fullBlocksPerGrid, blockSize>>>(n, dev_A, dev_B, depth);
				int* swap = dev_A;
				dev_A = dev_B;
				dev_B = swap;
			}
			kernExclusivePrefix<<<fullBlocksPerGrid, blockSize>>>(n, dev_A, dev_B);
            timer().endGpuTimer();

			// Get the return value off of the device and free memory.
			cudaMemcpy(odata, dev_A, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_A);
			cudaFree(dev_B);
        }
    }
}
