#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweepIteration(int n, int d, int *idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			k = k * (1 << (d + 1)) + (1 << (d + 1)) - 1;
			if (k > n || k < 0) { return; }

			int offset = 1 << d + 1;
			int old_val = idata[k];

			
			idata[k] = idata[k] + idata[k - (offset / 2)];

			//printf("d = %i, %i off %i %i oldval: %i val: %i\n", d, k, offset, offset / 2,old_val, idata[k]);
			
		}

		__global__ void kernDownSweepIteration(int n, int d, int *idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			k = k * (1 << (d + 1)) + (1 << (d + 1)) - 1;
			if (k > n || k < 0) { return; }

			int offset = 1 << d + 1;
			int add = idata[k] + idata[k - (offset / 2)];
			int replace = idata[k];
			int old_val = idata[k];
			idata[k - (offset / 2)] = replace;
			idata[k] = add;
			//printf("d = %i, %i off %i %i oldval: %i val: %i\n", d, k, offset, offset / 2, old_val, idata[k]);
			

		}

		__global__ void kernSetZero(int n, int *idata) {
			idata[n] = 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
			dim3 threadsPerBlock(128);
			int blockSize = 128;
			

			
			int arr_size = n;
			int log = ilog2ceil(arr_size);
			if ((n & (n - 1)) != 0) {
				arr_size = 1 << log;
			}
			
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, arr_size * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			int* dev_shiftdata;
			cudaMalloc((void**)&dev_shiftdata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_shiftdata failed!");

			cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			timer().startGpuTimer();
			//UPSWEEP
			log = ilog2ceil(arr_size);
			for (int d = 0; d < log; d++) {
				int off_n = arr_size / (1 << (d + 1));
				dim3 fullBlocksPerGrid((off_n + blockSize - 1) / blockSize);
				kernUpSweepIteration << <fullBlocksPerGrid, blockSize >> > (arr_size, d, dev_odata);
			}
			//DOWNSWEEP
			kernSetZero <<<1, 1 >>> (arr_size - 1, dev_odata);
			for (int d = log - 1; d >= 0; d--) {
				int off_n = arr_size / (1 << (d + 1));
				dim3 fullBlocksPerGrid((off_n + blockSize - 1) / blockSize);
				kernDownSweepIteration << <fullBlocksPerGrid, blockSize >> > (arr_size, d, dev_odata);
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy failed!");
			cudaFree(dev_odata);
			cudaFree(dev_shiftdata);
        }

		__global__ void kernMapToBoolean(int n, int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > n) { return; }
			int data = idata[index];
			if (data != 0) {
				idata[index] = 1;
			}
		}
		__global__ void kernScatter(int n, int *odata, int *idata, int *scanOutData, int *boolOutData) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > n) { return; }
			int boolIndex = boolOutData[index];
			if (boolIndex != 0) {
				odata[scanOutData[index]] = idata[index];
			}
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

			int blockSize = 128;
			dim3 threadsPerBlock(blockSize);

			int* dev_idata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			int* dev_boolData;
			cudaMalloc((void**)&dev_boolData, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMemcpy(dev_boolData, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			int* dev_scanData;
			cudaMalloc((void**)&dev_scanData, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			//timer().startGpuTimer();

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_boolData);

			int *boolOutData = new int[n];
			cudaMemcpy(boolOutData, dev_boolData, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy failed!");

			/*for (int i = 0; i < 10; i++) {
				printf("%i %i\n", i, boolOutData[i]);
			}*/

			int *scanOutData = new int[n];
			scan(n,scanOutData ,boolOutData);
			int compact_size = scanOutData[n-1] + boolOutData[n-1];
			cudaMemcpy(dev_scanData, scanOutData, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			kernScatter << <fullBlocksPerGrid, blockSize >> > (n,dev_odata, dev_idata, dev_scanData, dev_boolData);
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy failed!");
            //timer().endGpuTimer();
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_boolData);
			cudaFree(dev_scanData);
            return compact_size;
        }
    }
}
