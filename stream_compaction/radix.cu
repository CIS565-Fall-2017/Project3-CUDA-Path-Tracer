#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		/**
         * Maps an array to an array of 0s and 1s for bit n.
         */
        __global__ void kernMapToBoolean(int pow2n, int n, int k, int *bools, const int *idata) {
            // TODO
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= pow2n) return;

			int val = 1 << k;
			bools[index] = ((idata[index] & val) != 0 || index >= n) ? 0 : 1;
        }

        /**
         * Computes t array: t[i] = i - f[i] + totalFalses; f = idata, t = odata
         */
        __global__ void kernComputeT(int n, int totalFalses, int *odata, const int *idata) {
            // TODO
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			odata[index] = index - idata[index] + totalFalses;
        }

        /**
         * Computes d array: d[i] = e[i] ? f[i] : t[i]; 
         */
        __global__ void kernComputeD(int n, int *e, const int *f, const int *t) {
            // TODO
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			e[index] = (e[index] != 0) ? f[index] : t[index];
        }

        /**
         * Computes scattered array based on D indices
         */
        __global__ void kernScatterD(int n, int *odata, int *indices, const int *idata) {
            // TODO
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			odata[indices[index]] = idata[index];
        }

		void sort_implementation(int n, int k, int* last_e, int* last_f, 
								 int *e_arr, int *f_arr, int *t_arr, int *dev_in)
		{
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

			int pow2n = 1 << ilog2ceil(n);

			// Step 1: compute e array
			kernMapToBoolean<<<blocksPerGrid, blockSize>>>(pow2n, n, k, f_arr, dev_in);
			checkCUDAError("kernMapToBoolean failed!");

			cudaMemcpy(e_arr, f_arr, n * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpyDeviceToDevice failed!");

			// Step 2: exclusive scan e
			StreamCompaction::Efficient::scan_implementation(pow2n, f_arr);

			// Step 3: compute totalFalses
			cudaMemcpy(last_e, e_arr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("last_e cudaMemcpyDeviceToHost failed!");

			cudaMemcpy(last_f, f_arr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("last_f cudaMemcpyDeviceToHost failed!");

			int totalFalses = *last_e + *last_f;

			// Step 4: compute t array
			kernComputeT<<<blocksPerGrid, blockSize>>>(n, totalFalses, t_arr, f_arr);
			checkCUDAError("kernComputeT failed!");

			// Step 4: scatter based on address d
			kernComputeD<<<blocksPerGrid, blockSize>>>(n, e_arr, f_arr, t_arr);
			checkCUDAError("kernComputeD failed!");

			kernScatterD<<<blocksPerGrid, blockSize>>>(n, f_arr, e_arr, dev_in);
			checkCUDAError("kernScatterD failed!");
		}

        /**
         * Performs radix sort on idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        void sort(int n, int d, int *odata, const int *idata) 
		{
			// TODO
			int *e_arr;
			int *f_arr;
			int *t_arr;
			int *dev_in;

			int pow2n = 1 << ilog2ceil(n);

			int *last_e = (int *)malloc(sizeof(int));
			int *last_f = (int *)malloc(sizeof(int));

			cudaMalloc((void**)&e_arr, n * sizeof(int));
			checkCUDAError("cudaMalloc e_arr failed!");

			cudaMalloc((void**)&f_arr, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc f_arr failed!");

			cudaMalloc((void**)&t_arr, n * sizeof(int));
			checkCUDAError("cudaMalloc t_arr failed!");

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in failed!");

			timer().startGpuTimer();
			for (int k = 0; k < d; ++k) {
				sort_implementation(n, k, last_e, last_f, e_arr, f_arr, t_arr, dev_in);
				std::swap(dev_in, f_arr);
				/*cudaMemcpy(dev_in, f_arr, n * sizeof(int), cudaMemcpyDeviceToDevice);
				checkCUDAError("cudaMemcpyDeviceToDevice failed!");*/
			}
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			free(last_e);
			free(last_f);

			cudaFree(e_arr);
			cudaFree(f_arr);
			cudaFree(t_arr);
			cudaFree(dev_in);
        }
    }
}
