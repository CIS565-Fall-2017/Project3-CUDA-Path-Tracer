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

		/*
			Notes:

			This uses the "Naive" algorithm from GPU Gems 3, Section 39.2.1.
			Example 39-1 uses shared memory. This is not required in this project. You can simply use global memory.
			As a result of this, you will have to do ilog2ceil(n) separate kernel invocations.

			Since your individual GPU threads are not guaranteed to run simultaneously,
			you can't generally operate on an array in-place on the GPU; it will cause race conditions.
			Instead, create two device arrays. Swap them at each iteration:
			read from A and write to B, read from B and write to A, and so on.

			Beware of errors in Example 39-1 in the chapter;
			both the pseudocode and the CUDA code in the online version of Chapter 39
			are known to have a few small errors (in superscripting, missing braces, bad indentation, etc.)

			Be sure to test non-power-of-two-sized arrays.
		*/

        // TODO: __global__
		__global__ void computeNaiveScanHelper(int n, int factor, int *odata, const int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (index < n)
			{
				if (index >= factor)
				{
					odata[index] = idata[index - factor] + idata[index];
				}
				else
				{
					odata[index] = idata[index];
				}
			}
		}//end computeNaiveScanHelper


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
        void scan(int n, int *odata, const int *idata) {
			// TODO

			//create 2 device arrays
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int *inArray;
			int *outArray;
			cudaMalloc((void**)&inArray, n * sizeof(int));
			checkCUDAError("cudaMalloc inArray failed!");
			cudaMalloc((void**)&outArray, n * sizeof(int));
			checkCUDAError("cudaMalloc outArray failed!");
			cudaThreadSynchronize();	

			//Copy input data to GPU
			cudaMemcpy(inArray, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			for (int d = 1; d <= ilog2ceil(n); d++)
			{
				//Call kernel
				int factor = 1 << (d - 1);
				computeNaiveScanHelper<<<fullBlocksPerGrid, blockSize>>>(n, factor, outArray, inArray);

				//Make sure the GPU finishes before the next iteration of the loop
				cudaThreadSynchronize();

				//Swap arrays at each iteration
				std::swap(outArray, inArray);
			}

			timer().endGpuTimer();

			//Copy output data back to CPU
			//Make sure you're copying from inArray since you swap them every iteration
			odata[0] = 0;
			cudaMemcpy(odata + 1, inArray, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);

			//FREE THE ARRAYS
			cudaFree(inArray);
			cudaFree(outArray);
			checkCUDAError("cudaFree failed!");


        }//end scan function 
    }//end namespace Naive
}//end namespace StreamCompaction
