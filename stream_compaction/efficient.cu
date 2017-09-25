#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 1024

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernComputeUpSweepIteration(int n, int d, int *data) {
			//Compute the indices to be added together based on the thread index
			//and the iteration number. The second of these indices is also the output
			//index.
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index >= n) {
				return;
			}
			int incrementLength = 1 << d;
			int firstIndexInIteration = incrementLength - 1;
			int firstSumIndex = firstIndexInIteration + index * incrementLength * 2;			 
			data[firstSumIndex + incrementLength] = data[firstSumIndex] + data[firstSumIndex + incrementLength];
		}

		__global__ void kernComputeUpSweepIterationWithSharedMem(int n, int d, int *data) {
			//Compute the indices to be added together based on the thread index
			//and the iteration number. The second of these indices is also the output
			//index.
			__shared__ float temp[blockSize];
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			int threads = n;
			if (index >= threads) {
				return;
			}
			//load into shared memory
			temp[threadIdx.x] = data[index];
			__syncthreads();

			int incrementLength = 1;
			int firstIndexInIteration = incrementLength - 1;
			int firstSumIndex = firstIndexInIteration + index * incrementLength * 2;
			temp[threadIdx.x] = temp[firstSumIndex] + temp[firstSumIndex + incrementLength];
			//subsequent iterations
			for (int iter = 1; iter < d; iter++) {
				__syncthreads();
				threads /= 2;
				if (index < threads) {
					incrementLength << 1;
					firstIndexInIteration = incrementLength - 1;
					firstSumIndex = firstIndexInIteration + index * incrementLength * 2;
					temp[threadIdx.x] = temp[firstSumIndex] + temp[firstSumIndex + incrementLength];
				}
			}
			__syncthreads();
			data[index] = temp[threadIdx.x];
		}

		__global__ void kernComputeDownSweepIteration(int n, int d, int ceil, int *data) {
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index >= n) {
				return;
			}
			float twoPower = 1.0f / (float) (1 << d + 1);
			int incrementLength = ceil * twoPower;
			int rightIndex = (ceil - 1) - index * incrementLength * 2;

			int tmp = data[rightIndex - incrementLength];
			int dataFromRight = data[rightIndex];

			data[rightIndex - incrementLength] = dataFromRight;
			data[rightIndex] = dataFromRight + tmp;
		}

		__global__ void kernReplaceIndexWithZero(int n, int *data) {
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index != 0) {
				return;
			}
			data[n] = 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			//allocate device data padded with zeroes
			int log = ilog2ceil(n);
			int ceil = 1 << log;
			int *dev_data;
			cudaMalloc((void**)&dev_data, ceil * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");

			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

            timer().startGpuTimer();
			
			int numThreadsToDoWork = ceil;
			for (int d = 0; d < log; d++) {
				numThreadsToDoWork /= 2;
				dim3 fullBlocks((numThreadsToDoWork + blockSize - 1) / blockSize);
				kernComputeUpSweepIteration << <fullBlocks, blockSize >> > (numThreadsToDoWork, d, dev_data);
			}
			//kernComputeUpSweepIterationWithSharedMem << <1, blockSize >> > (numThreadsToDoWork, log, dev_data);

			kernReplaceIndexWithZero << <1, 32 >> > (ceil - 1, dev_data);
			numThreadsToDoWork = 1;
			for (int d = 0; d < log; d++) {
				dim3 fullBlocks((numThreadsToDoWork + blockSize - 1) / blockSize);
				kernComputeDownSweepIteration << <fullBlocks, blockSize >> > (numThreadsToDoWork, d, ceil, dev_data);
				numThreadsToDoWork *= 2;
			}
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");
			cudaFree(dev_data);
        }

		void scanInPlace(int n, int *dev_data) {
			int log = ilog2ceil(n);
			int ceil = (int)powf(2.0f, log);
			int numThreadsToDoWork = ceil;
			for (int d = 0; d < log; d++) {
				numThreadsToDoWork /= 2;
				dim3 fullBlocks((numThreadsToDoWork + blockSize - 1) / blockSize);
				kernComputeUpSweepIteration << <fullBlocks, blockSize >> > (numThreadsToDoWork, d, dev_data);
			}

			kernReplaceIndexWithZero << <1, 32 >> > (ceil - 1, dev_data);

			for (int d = 0; d < log; d++) {
				dim3 fullBlocks((numThreadsToDoWork + blockSize - 1) / blockSize);
				kernComputeDownSweepIteration << <fullBlocks, blockSize >> > (numThreadsToDoWork, d, ceil, dev_data);
				numThreadsToDoWork *= 2;
			}
		}

		/**
		Helper function to calculate how many elements made it through compaction
		*/

		__global__ void kernComputeNumberOfValidElements(int n, const int *bools, const int *indices, int *answer) {
			int index = threadIdx.x + blockDim.x * blockIdx.x; 
			if (index != 0) {
				return;
			}
			*answer = (bools[n] == 0) ? indices[n]: indices[n] + 1;
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
			int *dev_bools;
			int *dev_indices;
			int *dev_idata;
			int *dev_odata;
			int *dev_answer;

			int log = ilog2ceil(n);
			int ceil = 1 << log;

			cudaMalloc((void**)&dev_bools, ceil * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");

			cudaMalloc((void**)&dev_indices, ceil * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");
			
			cudaMalloc((void**)&dev_idata, ceil * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_odata, ceil * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_answer, sizeof(int));
			checkCUDAError("cudaMalloc dev_answer failed!");

			dim3 fullBlocks((ceil + blockSize - 1) / blockSize);

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

            timer().startGpuTimer();
            // map the idata to a bools array
			Common::kernMapToBoolean << <fullBlocks, blockSize >> > (ceil, dev_bools, dev_idata);
			

			cudaMemcpy(dev_indices, dev_bools, ceil * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy failed!");

			// perform scan on the bools array
			scanInPlace(n, dev_indices);

			Common::kernScatter << <fullBlocks, blockSize >>> (ceil, dev_odata, dev_idata, dev_bools, dev_indices);

			kernComputeNumberOfValidElements << <1, 32 >> > (ceil-1, dev_bools, dev_indices, dev_answer);
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			int answer;
			int *answerPtr = &answer;
			cudaMemcpy(answerPtr, dev_answer, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");


			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_answer);

			return answer;
        }
    }
}
