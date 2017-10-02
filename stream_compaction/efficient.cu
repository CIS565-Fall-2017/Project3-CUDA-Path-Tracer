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
		/*****************
		* Configuration *
		*****************/

		/*! Block size used for CUDA kernel launch. */
		#define blockSize 128

		// Implementation of a level's step in scan. Slide 43
		// Find the location of the previous sum and add it to its new position.
		__global__ void kernUpsweep(int numObjects, int* odata, int level) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= numObjects) {
				return;
			}

			// Every add in this operation is shifted right based on the depth of the tree.
			int sumIndexShift = powf(2, level);

			// Find which element the current thread is supposed to access.
			int currentIndex = (index + 1) * (2 * sumIndexShift) - 1;

			// Add the element which is shifted to the left.
			// This is equivalent to finding the left child of the parent node.
			// The current element is the right child; add to update the array.
			odata[currentIndex] += odata[currentIndex - sumIndexShift];
		}

		// Make this an exclusive prefix scan by making the first element 0.
		// Push all other elements back. Slide 46
		__global__ void kernDownsweep(int numObjects, int* odata, int level) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= numObjects) {
				return;
			}

			// Every add in this operation is shifted left based on the depth of the tree.
			int sumIndexShift = powf(2, level);

			// Find which element the current thread is supposed to access.
			int currentIndex = (index + 1) * (2 * sumIndexShift) - 1;

			// Perform the actual addition.
			int temp = odata[currentIndex - sumIndexShift];
			odata[currentIndex - sumIndexShift] = odata[currentIndex];
			odata[currentIndex] += temp;
		}

		// This kernel is used to zero the n-1th element, the root, for downsweeping.
		__global__ void kernZeroRoot(int n, int *idata) {
			idata[n] = 0;
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scanNoTimer(int n, int *odata, const int *idata) {
			/*
			Since the work-efficient scan operates on a binary tree structure, it
			works best with arrays with power-of-two length. Make sure your
			implementation works on non-power-of-two sized arrays (see ilog2ceil).
			This requires extra memory:
			your intermediate array sizes will need to be rounded to the next
			power of two.
			*/
			int nextPowerOfTwo = 1;
			while (nextPowerOfTwo < n) {
				nextPowerOfTwo *= 2;
			}

			// "Create two device arrays. Swap them at each iteration: 
			// read from A and write to B, read from B and write to A."
			int* dev_A;
			cudaMalloc((void**)&dev_A, nextPowerOfTwo * sizeof(int));
			// Zero out this chunk of memory since it's not getting filled by idata
			cudaMemset(dev_A, 0, nextPowerOfTwo * sizeof(int));
			// How to copy data to the GPU
			cudaMemcpy(dev_A, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			// Build the summed values before downsweeping
			int depthMax = ilog2ceil(nextPowerOfTwo);
			for (int depth = 0; depth < depthMax; ++depth) {
				dim3 fullBlocksPerGrid(((nextPowerOfTwo / pow(2, depth + 1)) + blockSize - 1) / blockSize);
				kernUpsweep<<<fullBlocksPerGrid, blockSize>>>(nextPowerOfTwo, dev_A, depth);
			}

			// "Set root to zero"
			dim3 fullBlocksPerGrid((nextPowerOfTwo + blockSize - 1) / blockSize);
			kernZeroRoot<<<fullBlocksPerGrid, blockSize>>>(nextPowerOfTwo - 1, dev_A);

			// "Traverse back down tree using partial sums to build the scan"
			for (int depth = (depthMax - 1); depth > -1; --depth) {
				dim3 fullBlocksPerGrid(((nextPowerOfTwo / pow(2, depth + 1)) + blockSize - 1) / blockSize);
				kernDownsweep<<<fullBlocksPerGrid, blockSize>>>(nextPowerOfTwo, dev_A, depth);
			}

			// Get the return value off of the device and free memory.
			cudaMemcpy(odata, dev_A, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_A);
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			/*
			Since the work-efficient scan operates on a binary tree structure, it 
			works best with arrays with power-of-two length. Make sure your 
			implementation works on non-power-of-two sized arrays (see ilog2ceil). 
			This requires extra memory:
				your intermediate array sizes will need to be rounded to the next
				power of two.
			*/
			int nextPowerOfTwo = 1;
			while (nextPowerOfTwo < n) {
				nextPowerOfTwo *= 2;
			}

			// Do not time memory allocation
			// "Create two device arrays. Swap them at each iteration: 
			// read from A and write to B, read from B and write to A."
			int* dev_A;
			cudaMalloc((void**)&dev_A, nextPowerOfTwo * sizeof(int));
			// Zero out this chunk of memory since it's not getting filled by idata
			cudaMemset(dev_A, 0, nextPowerOfTwo * sizeof(int));
			// How to copy data to the GPU
			cudaMemcpy(dev_A, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			// Time everything else
			timer().startGpuTimer();

			// Build the summed values before downsweeping
			int depthMax = ilog2ceil(nextPowerOfTwo);
			for (int depth = 0; depth < depthMax; ++depth) {
				dim3 fullBlocksPerGrid(((nextPowerOfTwo / pow(2, depth + 1)) + blockSize - 1) / blockSize);
				kernUpsweep<<<fullBlocksPerGrid, blockSize>>>(nextPowerOfTwo, dev_A, depth);
				checkCUDAErrorFn("upsweep failed");
			}

			// "Set root to zero"
			dim3 fullBlocksPerGrid((nextPowerOfTwo + blockSize - 1) / blockSize);
			kernZeroRoot<<<fullBlocksPerGrid, blockSize>>>(nextPowerOfTwo - 1, dev_A);
			checkCUDAErrorFn("zeroRoot failed");

			// "Traverse back down tree using partial sums to build the scan"
			for (int depth = (depthMax - 1); depth > -1; --depth) {
				dim3 fullBlocksPerGrid(((nextPowerOfTwo / pow(2, depth + 1)) + blockSize - 1) / blockSize);
				kernDownsweep<<<fullBlocksPerGrid, blockSize>>>(nextPowerOfTwo, dev_A, depth);
				checkCUDAErrorFn("downsweep failed");
			}
			timer().endGpuTimer();

			// Get the return value off of the device and free memory.
			cudaMemcpy(odata, dev_A, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_A);
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
			// Do not time memory allocation
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			int* dev_idata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			int* dev_bools;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			int* dev_indices;
			cudaMalloc((void**)&dev_indices, n * sizeof(int));

			// Time everything else
			timer().startGpuTimer();
			
			// Step 1: Compute temporary array.
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
			
			// Start counting the number of elements--check this last index in dev_bools to see if
			// it contains the element which might get shifted out after scanning.
			int elementsRemaining;
			cudaMemcpy(&elementsRemaining, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

			// Step 2: Run exclusive scan on temporary array
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
			scanNoTimer(n, dev_indices, dev_indices);

			// Get this final sum from scanning and add it to the variable already accounting for the
			// potentially-shifted element.
			int elementsRemainingAdd;
			cudaMemcpy(&elementsRemainingAdd, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			elementsRemaining += elementsRemainingAdd;

			// Step 3: Scatter
			Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

			// Get the return value off of the device and free memory.
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
            return elementsRemaining;
        }
    }
}
