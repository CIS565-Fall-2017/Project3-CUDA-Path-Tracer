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
		__global__ void kernScanByLevel(const int n, const int offset, int* odata, const int* idata) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			if (index >= offset) {
				odata[index] = idata[index] + idata[index - offset];
			} else { //final result already found for this position
				odata[index] = idata[index];
			}
		}

		__global__ void kernConvertToExclusiveScan(const int n, int* odata, const int* idata) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			if (index == 0) {
				odata[index] = 0;
			} else {
				odata[index] = idata[index - 1];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(const int n, int *odata, const int *idata) {
			const int numbytes = n * sizeof(int);

			int* dev_odata;
			cudaMalloc((void**)&dev_odata, numbytes);
			checkCUDAError("cudaMalloc dev_odata failed!");

			int* dev_idata;
			cudaMalloc((void**)&dev_idata, numbytes);
			checkCUDAError("cudaMalloc dev_idata failed!");


			cudaMemcpy(dev_idata, idata, numbytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_idata failed!");

			const int gridDim = (n + blockSize - 1) / blockSize;
			const int blockDim = blockSize;

            timer().startGpuTimer();
			for (int offset = 1; offset < n; offset <<= 1) {
				//gridDims.x can probably = (n + blockSize - 1 - offset) / blockSize;
				kernScanByLevel<<<gridDim, blockDim>>>(n, offset, dev_odata, dev_idata);
				std::swap(dev_odata, dev_idata);
			}

			//result is inclusive scan (includes the final reduction sum) 
			//shift left and odata[0] = 0 to get exclusive scan (identity at index 0 and remove final reduction sum)
			kernConvertToExclusiveScan<<<gridDim, blockDim>>>(n, dev_odata, dev_idata);

            timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, numbytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata to odata failed!");
			
			cudaFree(dev_idata);
			checkCUDAError("cudaFree of dev_idata failed!");

			cudaFree(dev_odata);
			checkCUDAError("cudaFree of dev_odata failed!");
        }
    }
}