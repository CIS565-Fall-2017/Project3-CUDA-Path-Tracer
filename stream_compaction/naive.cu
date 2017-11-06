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
        // TODO
		__global__ void kernNaiveScan(int n, int val, int *odata, int *idata) 
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;
			odata[index] = (index >= val) ? idata[index] + idata[index - val] : idata[index];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
			int *dev_in;
			int *dev_out;

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in failed!");
			cudaMemcpy(dev_out, odata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_out failed!");

			timer().startGpuTimer();
			for (int d = 1; d <= ilog2ceil(n); ++d) {
				int val = 1 << (d - 1);
				kernNaiveScan << <blocksPerGrid, blockSize >> > (n, val, dev_out, dev_in);
				std::swap(dev_in, dev_out);
				checkCUDAError("kernNaiveScan failed!");
			}

			timer().endGpuTimer();

			cudaMemcpy(odata + 1, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dev_in);
			cudaFree(dev_out);
        }
    }
}
