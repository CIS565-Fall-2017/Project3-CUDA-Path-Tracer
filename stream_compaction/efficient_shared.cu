#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient_Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernScanShared(int n, int *odata, int *sumOfSums)
		{
			extern __shared__ int temp[];
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;
			int offset = 1;

			// load shared memory
			int idx = threadIdx.x;
			//int bankOffset = conflictFreeOffset(idx);
			int bankOffset = 0;
			temp[idx + bankOffset] = odata[index];
			//temp[2 * index + 1] = odata[2 * index + 1];

			// upsweep
			for (int d = blockSize >> 1; d > 0; d >>= 1) {
				__syncthreads();
				if (idx < d) {
					int ai = offset * (2 * idx + 1) - 1;
					int bi = offset * (2 * idx + 2) - 1;
					ai += conflictFreeOffset(ai);
					bi += conflictFreeOffset(bi);

					temp[bi] += temp[ai];
				}
				offset *= 2;
			}

			if (idx == 0) {
				sumOfSums[blockIdx.x] = temp[blockDim.x - 1 + conflictFreeOffset(blockDim.x - 1)];
				temp[blockDim.x - 1 + conflictFreeOffset(blockDim.x - 1)] = 0;
			}

			// downsweep
			for (int d = 1; d < blockSize; d *= 2) {
				offset >>= 1;
				__syncthreads();
				if (idx < d) {
					int ai = offset * (2 * idx + 1) - 1;
					int bi = offset * (2 * idx + 2) - 1;
					ai += conflictFreeOffset(ai);
					bi += conflictFreeOffset(bi);

					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			
			__syncthreads();

			odata[index] = temp[idx + bankOffset];
			//odata[2 * index + 1] = temp[2 * index + 1];
		}

		__global__ void kernAddSums(int n, int *odata, int *sumOfSums)
		{
			__shared__ int sum;
			if (threadIdx.x == 0) sum = sumOfSums[blockIdx.x];
			__syncthreads();
			
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			odata[index] += sum;
		}

		/**
		* invariant: n must be power-of-2
		*/
		void scan_implementation(int pow2n, int block2n, int *dev_out, int *sumOfSums) 
		{
			int numBlocks1 = (pow2n + blockSize - 1) / blockSize;
			int numBlocks2 = (block2n + blockSize - 1) / blockSize;

			kernScanShared << <numBlocks1, blockSize, sizeof(int) * blockSize >> > (pow2n, dev_out, sumOfSums);
			checkCUDAError("kernScanShared 1 failed!");

			kernScanShared << <numBlocks2, blockSize, sizeof(int) * blockSize >> > (block2n, sumOfSums, sumOfSums);
			checkCUDAError("kernScanShared 2 failed!");

			kernAddSums << <numBlocks1, blockSize >> > (pow2n, dev_out, sumOfSums);
			checkCUDAError("kernAddSums failed!");
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata)
		{
			int *dev_out;
			int *sumOfSums;
			int pow2n = 1 << ilog2ceil(n);
			int block2n = 1 << ilog2ceil(pow2n / blockSize);

			cudaMalloc((void**)&dev_out, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			cudaMalloc((void**)&sumOfSums, block2n * sizeof(int));
			checkCUDAError("cudaMalloc sumOfSums failed!");

			cudaMemset(dev_out, 0, pow2n * sizeof(int));
			checkCUDAError("cudaMemset dev_out failed!");

			cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_out failed!");

			timer().startGpuTimer();
			scan_implementation(pow2n, block2n, dev_out, sumOfSums);
			timer().endGpuTimer();
			checkCUDAError("scan_implementation failed!");

			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dev_out);
			cudaFree(sumOfSums);
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
        int compact(int n, int *odata, const int *idata) 
		{
			// TODO
			int *dbools;
			int *dev_in;
			int *dev_out;
			int *sumOfSums;

			int pow2n = 1 << ilog2ceil(n);
			int block2n = 1 << ilog2ceil(pow2n / blockSize);

			cudaMalloc((void**)&dbools, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc dbools failed!");

			cudaMemset(dbools, 0, pow2n * sizeof(int));
			checkCUDAError("cudaMemset dbools failed!");

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in failed!");

			cudaMalloc((void**)&sumOfSums, block2n * sizeof(int));
			checkCUDAError("cudaMalloc sumOfSums failed!");

			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			timer().startGpuTimer();

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (n, dbools, dev_in);
			checkCUDAError("kernMapToBoolean failed!");

			int num;
			cudaMemcpyAsync(&num, dbools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			int ret = num;

			scan_implementation(pow2n, block2n, dbools, sumOfSums); // requires power of 2

			cudaMemcpyAsync(&num, dbools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");
			ret += num;

			StreamCompaction::Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_out, dev_in, dbools);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_out, ret * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dbools);
			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(sumOfSums);
            
            return ret;
        }
    }
}
