#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 512
//#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernEfficientUpsweep(int pow2plus1, int pow2, int N, int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < N) {
				if (index % pow2plus1==0) {
					idata[index + pow2plus1 - 1] += idata[index + pow2 - 1];
				}
			}
		}

		__global__ void kernEfficientDownsweep(int pow2plus1, int pow2, int N, int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < N) {
				if (index % pow2plus1 == 0) {
					int t = idata[index + pow2 - 1];
					idata[index + pow2 - 1] = idata[index + pow2plus1 - 1];
					idata[index + pow2plus1 - 1] += t;
				}
			}
		}

		__global__ void kernNonPowerTwoHelper(int N, int zeroStartIndex, int zeroEndIndex, int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < N) {
				if (index >= zeroStartIndex && index < zeroEndIndex) {
					idata[index] = 0;
				}
			}
		}

		__global__ void kernChangeElem(int *arr, int idx, int val) {
			arr[idx] = val;
		}
		
		void gpuEfficientScan(int originSize, int npoweroftwo, int* dev_tempin) {
			dim3 fullBlocksPerGrid((npoweroftwo + blockSize - 1) / blockSize);
			//Helper kernal function to set extra elements to 0 if the size is not a power of 2.
			if (originSize != npoweroftwo) {
				kernNonPowerTwoHelper << <fullBlocksPerGrid, blockSize >> > (npoweroftwo, originSize, npoweroftwo, dev_tempin);
			}
			
			for (int iteration = 0; iteration <= ilog2ceil(npoweroftwo) - 1; iteration++) {
				kernEfficientUpsweep << <fullBlocksPerGrid, blockSize >> > (pow(2, iteration + 1),
					pow(2, iteration), npoweroftwo, dev_tempin);
			}
			kernChangeElem << <1, 1 >> >(dev_tempin, npoweroftwo - 1, 0);

			for (int iteration = ilog2ceil(npoweroftwo) - 1; iteration >= 0; iteration--) {
				kernEfficientDownsweep << <fullBlocksPerGrid, blockSize >> > (pow(2, iteration + 1),
					pow(2, iteration), npoweroftwo, dev_tempin);
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			
			int *dev_tempin;
			//check if n is power of 2, if not, needs to add extra zeros in order to make n a power of 2
			bool isPowerOfTwo = (n != 0) && ((n & (n - 1)) == 0);
			int npoweroftwo = n;
			if (!isPowerOfTwo) {
				npoweroftwo = pow(2, ilog2ceil(n));
			}
			
			cudaMalloc((void**)&dev_tempin, npoweroftwo * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_tempin failed!");
			cudaMemcpy(dev_tempin, idata, npoweroftwo * sizeof(int), cudaMemcpyHostToDevice);
			//checkCUDAErrorWithLine("cuda Memcpy from idata to dev_tempin failed!");

			timer().startGpuTimer();
			gpuEfficientScan(n, npoweroftwo, dev_tempin);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_tempin, npoweroftwo * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorWithLine("cuda Memcpy from dev_tempin to odata failed!");
			cudaFree(dev_tempin);			
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
           
            // TODO
			bool isPowerOfTwo = (n != 0) && ((n & (n - 1)) == 0);
			int npoweroftwo = n;
			if (!isPowerOfTwo) {
				npoweroftwo = pow(2, ilog2ceil(n));
			}

			dim3 fullBlocksPerGrid((npoweroftwo + blockSize - 1) / blockSize);

			//Initialize two host arrays to store intermediate bools and indices
			int* ibools;
			ibools = (int*)malloc(npoweroftwo * sizeof(int));
			int* indices;
			indices = (int*)malloc(npoweroftwo * sizeof(int));
			
			//Initialize and allocate CUDA device arrays
			int *dev_bool;
			int *dev_idata;
			int *dev_odata;
			int *dev_indices;
			cudaMalloc((void**)&dev_bool, npoweroftwo * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_bool failed!");
			cudaMalloc((void**)&dev_idata, npoweroftwo * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, npoweroftwo * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_indices, npoweroftwo * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_indices failed!");
			
			//Copy data from host input data to device data
			cudaMemcpy(dev_idata, idata, npoweroftwo * sizeof(int), cudaMemcpyHostToDevice);
			//checkCUDAErrorWithLine("cuda Memcpy from idata to dev_idata failed!");

			//Perform a basic check in kernal, mark 1 for numbers not equal to 0, 0 for numbers equal to 0
			timer().startGpuTimer();
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (npoweroftwo, dev_bool, dev_idata);
			cudaDeviceSynchronize();

			//Copy result from device array dev_bool to device array dev_indices for excluse scan.
			cudaMemcpy(dev_indices, dev_bool, npoweroftwo * sizeof(int), cudaMemcpyDeviceToDevice);
			
			//Perform an exclusive sum scan on ibools to get the final indices array
			gpuEfficientScan(n, npoweroftwo, dev_indices);

			//perform the final scatter step to store the result in dev_odata
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bool, dev_indices);
            timer().endGpuTimer();

			//copy result from dev_odata to odata.
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorWithLine("cuda Memcpy from dev_odata to odata failed!");

			cudaFree(dev_bool);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_indices);

			//How to decide the remianing number in odata?
			int count=0;
			for (int i = 0; i < n; i++) {
				if (odata[i] != 0) {
					count++;
				}
				else {
					break;
				}
			}
            return count;
        }
    }
}
