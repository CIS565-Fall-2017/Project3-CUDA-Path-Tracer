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

		__global__ void kernZeroed(int totalN, int n, int *odata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= totalN) return;

			odata[index] = (index < n) ? odata[index] : 0;
		}

		__global__ void kernUpSweep(int n, int d, int *odata) 
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int offset = 1 << (d + 1);
			if (index >= n >> (d + 1)) return;
			int i = (index + 1) * offset - 1;
			if (i >= n) return;

			int val = 1 << d;
			odata[i] += odata[i - val];
			if (i == n - 1) odata[i] = 0;
		}

		__global__ void kernDownSweep(int n, int d, int *odata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int offset = 1 << (d + 1);
			if (index >= n >> (d + 1)) return;
			int i = (index + 1) * offset - 1;
			if (i >= n) return;

			int val = 1 << d;
			
			int temp = odata[i];
			odata[i] += odata[i - val];
			odata[i - val] = temp;
		}

		void scan_implementation(int pow2n, int *dev_out)
		{
			for (int d = 0; d < ilog2ceil(pow2n); ++d) {
				dim3 blocksPerGrid((pow2n / (1 << (d + 1)) + blockSize - 1) / blockSize);
				kernUpSweep << <blocksPerGrid, blockSize >> > (pow2n, d, dev_out);
				checkCUDAError("kernUpSweep failed!");
			}

			for (int d = ilog2ceil(pow2n) - 1; d >= 0; --d) {
				dim3 blocksPerGrid((pow2n / (1 << (d + 1)) + blockSize - 1) / blockSize);
				kernDownSweep << <blocksPerGrid, blockSize >> > (pow2n, d, dev_out);
				checkCUDAError("kernDownSweep failed!");
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
		{
			int *dev_out;
			int pow2n = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dev_out, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_out failed!");

			dim3 blocksPerGrid((pow2n + blockSize - 1) / blockSize);
			kernZeroed << <blocksPerGrid, blockSize >> > (pow2n, n, dev_out);
			checkCUDAError("kernZeroed failed!");
			
			timer().startGpuTimer();
			scan_implementation(pow2n, dev_out);
			timer().endGpuTimer();
			checkCUDAError("scan_implementation failed!");

			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dev_out);
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
        int compact(int n, int *odata, const int *idata) 
		{
			// TODO
			int *dbools;
			int *dev_in;
			int *dev_out;

			int pow2n = 1 << ilog2ceil(n);
			int block2n = 1 << ilog2ceil(pow2n / blockSize);

			cudaMalloc((void**)&dbools, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc dbools failed!");

			cudaMemset(dbools, 0, pow2n * sizeof(int));
			checkCUDAError("cudaMemset dbools failed!");

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in failed!");

			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			timer().startGpuTimer();

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (n, dbools, dev_in);
			checkCUDAError("kernMapToBoolean failed!");

			int num;
			cudaMemcpyAsync(&num, dbools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			int ret = num;

			scan_implementation(pow2n, dbools); // requires power of 2

			cudaMemcpyAsync(&num, dbools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");
			ret += num;

			StreamCompaction::Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_out, dev_in, dbools);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_out, ret * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dbools);
			cudaFree(dev_in);
			cudaFree(dev_out);

			return ret;
        }
    }
}
