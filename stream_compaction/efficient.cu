#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blocksize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void upSweep(int n, int pow2dPlus1, int pow2d, int *odata, bool reachedRoot)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			if (reachedRoot) {
				odata[n - 1] = 0;
			}
			else {
				index *= pow2dPlus1;
				if (index < n)
					odata[index + pow2dPlus1 - 1] += odata[index + pow2d - 1];
			}
		}

		__global__ void downSweep(int n, int pow2dPlus1, int pow2d, int *odata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			index *= pow2dPlus1;
			if (index < n) {
				int t = odata[index + pow2d - 1];
				odata[index + pow2d - 1] = odata[index + pow2dPlus1 - 1];
				odata[index + pow2dPlus1 - 1] += t;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 fullBlocksPerGrid((n + blocksize - 1) / blocksize);
			
			// Get the next power of 2
			int currPow = ilog2ceil(n) - 1;
			int nextPow = 2 << currPow;

			int *temp = new int[nextPow];
			for (int i = 0; i < nextPow; i++) {
				if (i < n) {
					temp[i] = idata[i];
				}
				// Fill the rest of the array with 0 if not a power of 2.
				else {
					temp[i] = 0;
				}
			}

			int *out;
			cudaMalloc((void**)&out, nextPow * sizeof(int));
			checkCUDAError("cudaMalloc out failed!");
			cudaMemcpy(out, temp, sizeof(int) * nextPow, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
            // TODO

			// Up-Sweep
			for (int d = 0; d <= ilog2ceil(nextPow) - 1; d++) {
				int pow2dPlus1 = pow(2, d + 1);
				int pow2d = pow(2, d);

				// If we hit the end of the depth then we should be writing to the very last spot in the array.
				bool reachedRoot = (d == ilog2ceil(nextPow) - 1);
				upSweep << < fullBlocksPerGrid, blocksize >> > (nextPow, pow2dPlus1, pow2d, out, reachedRoot);
			}

			// Down-Sweep
			for (int d = ilog2ceil(nextPow) - 1; d >= 0; d--) {
				int pow2dPlus1 = pow(2, d + 1);
				int pow2d = pow(2, d);
			
				downSweep << < fullBlocksPerGrid, blocksize >> > (nextPow, pow2dPlus1, pow2d, out);
			}

            timer().endGpuTimer();

			// Copy final values into odata
			cudaMemcpy(odata, out, sizeof(int) * nextPow, cudaMemcpyDeviceToHost);

			delete[]temp;
			cudaFree(out);
        }

		void exclusiveScan(int n, int *odata, const int *idata) {
			dim3 fullBlocksPerGrid((n + blocksize - 1) / blocksize);

			// Get the next power of 2
			int currPow = ilog2ceil(n) - 1;
			int nextPow = 2 << currPow;

			int *temp = new int[nextPow];
			for (int i = 0; i < nextPow; i++) {
				if (i < n) {
					temp[i] = idata[i];
				}
				// Fill the rest of the array with 0 if not a power of 2.
				else {
					temp[i] = 0;
				}
			}

			int *out;
			cudaMalloc((void**)&out, nextPow * sizeof(int));
			checkCUDAError("cudaMalloc out failed!");
			cudaMemcpy(out, temp, sizeof(int) * nextPow, cudaMemcpyHostToDevice);

			// Up-Sweep
			for (int d = 0; d <= ilog2ceil(nextPow) - 1; d++) {
				//int pow2dPlus1 = pow(2, d + 1);
				//int pow2d = pow(2, d);

				int pow2dPlus1 = pow(2, d + 1);
				int pow2d = pow(2, d);

				// If we hit the end of the depth then we should be writing to the very last spot in the array.
				bool reachedRoot = (d == ilog2ceil(nextPow) - 1);
				upSweep << < fullBlocksPerGrid, blocksize >> > (nextPow, pow2dPlus1, pow2d, out, reachedRoot);
			}

			// Down-Sweep
			for (int d = ilog2ceil(nextPow) - 1; d >= 0; d--) {
				int pow2dPlus1 = pow(2, d + 1);
				int pow2d = pow(2, d);

				downSweep << < fullBlocksPerGrid, blocksize >> > (nextPow, pow2dPlus1, pow2d, out);
			}

			// Copy final values into odata
			cudaMemcpy(odata, out, sizeof(int) * nextPow, cudaMemcpyDeviceToHost);

			delete[]temp;
			cudaFree(out);
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
			dim3 fullBlocksPerGrid((n + blocksize - 1) / blocksize);

			// Get the next power of 2
			int currPow = ilog2ceil(n) - 1;
			int nextPow = 2 << currPow;

			int *dev_in;
			cudaMalloc((void**)&dev_in, sizeof(int) * nextPow);
			cudaMemcpy(dev_in, idata, sizeof(int) * nextPow, cudaMemcpyHostToDevice);

			int *dev_out;
			cudaMalloc((void**)&dev_out, sizeof(int) * nextPow);

			int *dev_indices;
			cudaMalloc((void**)&dev_indices, sizeof(int) * nextPow);

			int *bools = new int[nextPow];
			int *dev_bools;
			cudaMalloc((void**)&dev_bools, sizeof(int) * nextPow);

			timer().startGpuTimer();
			// TODO

			StreamCompaction::Common::kernMapToBoolean << < fullBlocksPerGrid, blocksize >> > (nextPow, dev_bools, dev_in);
			cudaMemcpy(bools, dev_bools, sizeof(int) * nextPow, cudaMemcpyDeviceToHost);

			exclusiveScan(nextPow, dev_indices, bools);

			StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, blocksize >> > (nextPow, dev_out, dev_in, dev_bools, dev_indices);

			timer().endGpuTimer();

			int numElements = 0;
			for (int i = 0; i < n; i++) {
				if (bools[i] == 1) {
					numElements++;
				}
			}

			cudaMemcpy(odata, dev_out, sizeof(int) * numElements, cudaMemcpyDeviceToHost);

			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(dev_indices);
			cudaFree(dev_bools);

			delete[]bools;

			return numElements;
		}
    }
}
