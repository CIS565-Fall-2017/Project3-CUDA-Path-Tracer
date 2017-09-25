#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernScanIteration(int n, int d, int *idata, int* odata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;	
			if (k > n) { return; }
			//printf("index %i start %i d %i\n", k, 2 << d, d);
			//int offset = ((d - 2) < 0) ? 1 : 2 << (d - 2);
			int offset = 1 << d - 1;
			if (k >= offset) {
				odata[k] = idata[k - offset] + idata[k];
			}
			else {
				odata[k] = idata[k];
			}
		}

		__global__ void kernShiftArray(int n, int *idata, int* odata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k > n) { return; }

			if (k == 0) { 
				odata[0] = 0; 
			}
			else {
				odata[k] = idata[k - 1];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 threadsPerBlock(128);
			int blockSize = 128;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int* tempA;
			int* tempB; 
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc tempA failed!");

			//cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			cudaMalloc((void**)&tempA, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc tempA failed!");
			cudaMalloc((void**)&tempB, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc tempB failed!");

			cudaMemcpy(tempA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			cudaMemcpy(tempB, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");

            timer().startGpuTimer();
			int log = ilog2ceil(n);
			for(int d = 1; d <= log; d++) {
				kernScanIteration <<<fullBlocksPerGrid, blockSize >>> (n, d, tempA, tempB);
				if (d == log) {
					kernShiftArray << <fullBlocksPerGrid, blockSize >> > (n, tempB, dev_odata);
					cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
					checkCUDAErrorWithLine("memcpy back failed!");
				}
				else {
					//cudaMemcpy(tempA, tempB, sizeof(int) * n, cudaMemcpyDeviceToDevice);
					//checkCUDAErrorWithLine("memcpy failed!");
					int * temp;
					temp = tempA;
					tempA = tempB;
					tempB = temp;
				}
			}
            timer().endGpuTimer();
			cudaFree(tempA);
			cudaFree(tempB);
			cudaFree(dev_odata);
        }
    }
}
