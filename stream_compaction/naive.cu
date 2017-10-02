#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#define blockSize 512
//#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// TODO: __global__
		__global__ void kernNaiveScan(int pow2minus1, int N, int* odata, const int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < N) {
				if (index >= pow2minus1) {
					int scanIndexHelper = index - pow2minus1;
					odata[index] = idata[scanIndexHelper] + idata[index];
				}
				else {
					odata[index] = idata[index];
				}
			}
		}

		__global__ void kernRightShift(int N, int* odata, int* idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index == 0) {
				odata[index] = 0;
				odata[index + 1] = idata[index];
			}else if (index < N-1) {
				odata[index + 1] = idata[index];
			}
		}
		
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			bool swap = false; //flag, true call kernNaiveScan(iteration, n, dev_tempin, dev_tempout). false call kernNaiveScan(iteration, n, dev_tempout, dev_tempin)
			
            // TODO
			int *dev_tempin;
			int *dev_tempout;
			cudaMalloc((void**)&dev_tempin, n * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_tempin failed!");
			cudaMalloc((void**)&dev_tempout, n * sizeof(int));
			cudaMemcpy(dev_tempin, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			
			timer().startGpuTimer();
			for(int iteration = 1; iteration<= ilog2ceil(n); iteration++){
				if (swap) {
					kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (pow(2, iteration-1), n, dev_tempin, dev_tempout);
				}
				else {
					kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (pow(2, iteration - 1), n, dev_tempout, dev_tempin);
				}
				swap = !swap;				
			}
			swap = !swap; //revert back to the last status
			if (swap) {
				kernRightShift << <fullBlocksPerGrid, blockSize >> >(n, dev_tempout, dev_tempin);
			}
			else {
				kernRightShift << <fullBlocksPerGrid, blockSize >> >(n, dev_tempin, dev_tempout);
			}
            timer().endGpuTimer();
			
			
			if (swap) {
				cudaMemcpy(odata, dev_tempout, n * sizeof(int), cudaMemcpyDeviceToHost);
				//checkCUDAErrorWithLine("cuda Memcpy from dev_tempin to odata failed!");
			}
			else {
				cudaMemcpy(odata, dev_tempin, n * sizeof(int), cudaMemcpyDeviceToHost);
				//checkCUDAErrorWithLine("cuda Memcpy from dev_tempout to odata failed!");
			}
			cudaFree(dev_tempin);
			cudaFree(dev_tempout);
        }
    }
}
