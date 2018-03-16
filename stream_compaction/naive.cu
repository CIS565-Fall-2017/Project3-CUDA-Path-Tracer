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
        
		// TODO: __global__
		__global__ void scan(int N, int power, int *dev_oDataArray, int *dev_iDataArray) {
		
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;
			
			dev_oDataArray[index] = (index >= power) ? dev_iDataArray[index - power] + dev_iDataArray[index] : dev_iDataArray[index];
		
		}

		__global__ void kernExclusiveFromInclusive(int N, int *dev_oDataArray, int *dev_iDataArray) {
		
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;

			dev_oDataArray[index] = (index == 0) ? 0 : dev_iDataArray[index - 1];
		
		}
        
		/**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        
		void scan(int n, int *odata, const int *idata) {
			
			// Defining the configuration of the kernel
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			int size = n * sizeof(int);

			// Creating array buffers on the device memory
			int *dev_oDataArray, *dev_iDataArray;
			cudaMalloc((void**)&dev_oDataArray, size);
			checkCUDAError("cudaMalloc dev_oDataArray failed!");
			cudaMalloc((void**)&dev_iDataArray, size);
			checkCUDAError("cudaMalloc dev_iDataArray failed!");

			// Copying array buffers from Host to Device
			cudaMemcpy(dev_iDataArray, idata, size, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_iDataArray failed!");

			timer().startGpuTimer();

			// TODO
			int dimension = ilog2ceil(n);
			for (int d = 1; d <= dimension; ++d) {
				// Power of 2^(d-1)
				int power = 1 << (d - 1);
				scan <<<fullBlocksPerGrid, threadsPerBlock>>> (n, power, dev_oDataArray, dev_iDataArray);
				checkCUDAError("scan kernel failed!");
			
				std::swap(dev_oDataArray, dev_iDataArray);
			}
			
			// Convert the output data array from inclusve to exclusive
			kernExclusiveFromInclusive << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_oDataArray, dev_iDataArray);

			timer().endGpuTimer();

			//Copying array buffers from Device to Host
			cudaMemcpy(odata, dev_oDataArray, size, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata failed!");

			// Freeing the device memory
			cudaFree(dev_oDataArray);
			cudaFree(dev_iDataArray);
        
		}
	
	}

}
