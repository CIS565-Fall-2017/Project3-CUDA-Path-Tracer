#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

dim3 threadsPerBlock(blockSize);

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void kernPerformScanIteration(int n, int d, int *odata, const int *idata) {
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index > n) {
				return;
			}
			int twoPower = (int)powf(2.0f, d);
			odata[index] = (index >= twoPower) ? idata[index - twoPower] + idata[index] : idata[index];
		}

		__global__ void kernShiftInclusiveToExclusive(int n, int *odata, const int *idata) {
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index > n) {
				return;
			}
			odata[index] = (index > 0) ? idata[index - 1] : 0;
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			//define blocks
			dim3 fullBlocks((n + blockSize - 1) / blockSize);
			int numIterations = ilog2ceil(n);

			//allocate buffers in global memory
			int *dev_idata;
			int *dev_odata;
			int *tmpBuffer;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc of dev_idata failed!");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc of dev_idata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

			//populate dev_odata for continuity (really for x0, so this is mostly unnecessary)
			//cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			//checkCUDAError("cudaMemcpy failed!");
			timer().startGpuTimer();

			for (int d = 0; d < numIterations; d++) {
				kernPerformScanIteration << <fullBlocks, blockSize >> > (n, d, dev_odata, dev_idata);
				tmpBuffer = dev_idata;
				dev_idata = dev_odata;
				dev_odata = tmpBuffer;
			}
			//At this point, our final array is in dev_idata
			kernShiftInclusiveToExclusive << <fullBlocks, blockSize >> > (n, dev_odata, dev_idata);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");
			
			//Free buffers
			cudaFree(dev_idata);
			cudaFree(dev_odata);

        }
    }
}
