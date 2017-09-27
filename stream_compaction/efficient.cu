#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		
		void __global__ kernScanUpSweep (const int N, const int powerv1, const int powerv2, int *dev_oDataArray) {
			
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;

			if (index % powerv1 != powerv1 - 1) return;
			dev_oDataArray[index] += dev_oDataArray[index - powerv2];

			// After UpSweep is over the last value in the array is given value 0 which will be useful in the DownSweep step
			if (index == N - 1) {
				dev_oDataArray[index] = 0;
			}
		
		}

		void __global__ kernScanDownSweep (const int N, const int powerv1, const int powerv2, int *dev_oDataArray) {
			
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;

			if (index % powerv1 != powerv1 - 1) return;
			int temp = dev_oDataArray[index - powerv2];
			dev_oDataArray[index - powerv2] = dev_oDataArray[index];
			dev_oDataArray[index] += temp;
		
		}

		void __global__ kernPaddArrayWithZero(const int N, const int paddedArrayLength, int *dev_oDataArray) {
			
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < N || index >= paddedArrayLength) return;
		
			dev_oDataArray[index] = 0;
		
		}

		void scanExcusivePrefixSum(int N, int dimension, dim3 fullBlocksPerGrid, dim3 threadsPerBlock, int *dev_oDataArray) {
			
			// Up Sweep Scan
			int powerv1, powerv2;
			for (int d = 0; d < dimension; ++d) {
				powerv1 = 1 << (d + 1);
				powerv2 = 1 << d;
				kernScanUpSweep << <fullBlocksPerGrid, threadsPerBlock >> > (N, powerv1, powerv2, dev_oDataArray);
			}

			// Down Sweep Scans
			for (int i = dimension - 1; i >= 0; --i) {
				powerv1 = 1 << (i + 1);
				powerv2 = 1 << i;
				kernScanDownSweep << <fullBlocksPerGrid, threadsPerBlock >> > (N, powerv1, powerv2, dev_oDataArray);
			}
		
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        
		void scan(int n, int *odata, const int *idata) {
			
			// Defining the configuration of the kernel
			int dimension = ilog2ceil(n);
			int paddedArrayLength = 1 << (dimension);
			int paddedArraySize = paddedArrayLength * sizeof(int);
			dim3 fullBlocksPerGrid((paddedArrayLength + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);
			int size = n * sizeof(int);

			// Creating array buffers on the device memory
			int *dev_oDataArray;
			cudaMalloc((void**)&dev_oDataArray, paddedArraySize);
			checkCUDAError("cudaMalloc for dev_oDataArray failed!");

			// Copying array buffers (Host to Device)
			cudaMemcpy(dev_oDataArray, idata, size, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy into dev_oDataArray failed!");

			// For the extra space in the padded array fill it with 0
			kernPaddArrayWithZero <<<fullBlocksPerGrid, threadsPerBlock>>> (n, paddedArrayLength, dev_oDataArray);

            timer().startGpuTimer();
			
			// TODO
			scanExcusivePrefixSum(paddedArrayLength, dimension, fullBlocksPerGrid, threadsPerBlock, dev_oDataArray);
			checkCUDAError("scanExcusivePrefixSum failed!");

			timer().endGpuTimer();

			//Copying array buffers (Device to Host)
			cudaMemcpy(odata, dev_oDataArray, size, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy into odata failed!");

			// Freeing the data buffer stored in device memory
			cudaFree(dev_oDataArray);
			checkCUDAError("cudaFree on dev_oDataArray failed!");
        
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
			
			// Defining the configuration of the kernel
			int dimension = ilog2ceil(n);
			int paddedArrayLength = 1 << (dimension);
			int paddedArraySize = paddedArrayLength * sizeof(int);
			dim3 fullBlocksPerGridPadded((paddedArrayLength + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);
			int size = n * sizeof(int);
			int fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			// Creating buffers on the device memory
			int *dev_oScanDataArray;
			cudaMalloc((void**)&dev_oScanDataArray, paddedArraySize);
			checkCUDAError("cudaMalloc for dev_oScanDataArray failed!");
			
			int *dev_oData;
			cudaMalloc((void**)&dev_oData, size);
			checkCUDAError("cudaMalloc for dev_oData failed!");

			int *dev_iData;
			cudaMalloc((void**)&dev_iData, size);
			checkCUDAError("cudaMalloc for dev_iData failed!");

			int *dev_boolIndexArray;
			cudaMalloc((void**)&dev_boolIndexArray, size);
			checkCUDAError("cudaMalloc for dev_boolIndexArray failed!");

			// Copying array buffers idata to dev_iData (Host to Device)
			cudaMemcpy(dev_iData, idata, size, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy into dev_iData failed!");
			
			// Initialize the bool array: For each index fill 1 in dev_boolIndexArray[index] if corrosponding value in dev_iData is non-zero otherwise fill 0 
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, threadsPerBlock>>> (n, dev_boolIndexArray, dev_iData);
			checkCUDAError("kernMapToBoolean failed!");

			// Copy buffer dev_boolIndexArray to buffer dev_oScanDataArray (Device to Device)
			cudaMemcpy(dev_oScanDataArray, dev_boolIndexArray, size, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy into dev_oScanDataArray failed!");

			// Padd the dev_oScanDataArray with zero's at the end
			kernPaddArrayWithZero << <fullBlocksPerGridPadded, threadsPerBlock >> > (n, paddedArrayLength, dev_oScanDataArray);
			checkCUDAError("kernPaddArrayWithZero failed!");

			timer().startGpuTimer();
			
			// TODO
			scanExcusivePrefixSum(paddedArrayLength, dimension, fullBlocksPerGridPadded, threadsPerBlock, dev_oScanDataArray);
			checkCUDAError("scanExcusivePrefixSum failed!");
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_oData, dev_iData, dev_boolIndexArray, dev_oScanDataArray);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();
            
			// Copying the data from dev_oData to odata (Device To Host)
			cudaMemcpy(odata, dev_oData, size, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy into odata failed!");

			// Getting the size of the number of elements that were filled (Device To Host)
			int valueAtIndexArrayEnd, valueAtBoolArrayEnd, totalSize;
			cudaMemcpy(&valueAtIndexArrayEnd, dev_oScanDataArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&valueAtBoolArrayEnd, dev_boolIndexArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			totalSize = valueAtBoolArrayEnd + valueAtIndexArrayEnd;

			// Freeing Cuda Memory
			cudaFree(dev_boolIndexArray);
			cudaFree(dev_iData);
			cudaFree(dev_oData);
			cudaFree(dev_oScanDataArray);
			checkCUDAError("cudaFree failed!");

			return totalSize;
        
		}
    
	}

}
