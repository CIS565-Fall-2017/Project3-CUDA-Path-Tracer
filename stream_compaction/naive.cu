#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

static int blockSize = 1024;
static dim3 blockNum;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void Parallel_Add(int n, int d, int *odata, const int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index >= n)
				return;

			int flag = 1 << (d - 1);
			if (index >= flag) {
				odata[index] = idata[index - flag] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}

		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
			if (n <= 0)
				return;
			int celllog = ilog2ceil(n);
			odata[0] = 0;

			int *dev_data[2];
			
			cudaMalloc((void**)&dev_data[0], n * sizeof(int));
			checkCUDAError("cudaMalloc dev_data[0] failed!");

			cudaMalloc((void**)&dev_data[1], n * sizeof(int));
			checkCUDAError("cudaMalloc dev_data[1] failed!");

			cudaMemcpy(dev_data[0], idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");
			
			timer().startGpuTimer();
			int out_index = 0;
			
			blockNum = (n + blockSize - 1) / blockSize;
			for (int d = 1; d <= celllog; d++) {
				out_index ^= 1;
				Parallel_Add << <blockNum, blockSize >> > (n, d, dev_data[out_index], dev_data[out_index ^ 1]);
			}
			timer().endGpuTimer();
			
			cudaMemcpy(odata + 1, dev_data[out_index], (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMalloc dev_data[1] failed!");

			cudaFree(dev_data[0]);
			cudaFree(dev_data[1]);
			checkCUDAError("cudaMalloc dev_data[1] failed!");

        }
    }
}
