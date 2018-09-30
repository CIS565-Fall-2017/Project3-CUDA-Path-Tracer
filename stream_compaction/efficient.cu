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

		__global__ void upsweep(int n, int k, int* dev)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			if ((index % (2 * k) == 0) && (index + (2 * k) <= n))
				dev[index + (2 * k) - 1] += dev[index + k - 1];
		}

		__global__ void downsweep(int n, int k, int* dev)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			if ((index % (2 * k) == 0) && (index + (2 * k) <= n))
			{
				int tmp = dev[index + k - 1];
				dev[index + k - 1] = dev[index + (2 * k) - 1];
				dev[index + (2 * k) - 1] += tmp;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
       void scan(int n, int *odata, int *idata) {

			int* dev;
			int potn = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dev, potn * sizeof(int));
			checkCUDAError("Malloc for input device failed\n");

			cudaMemset(dev, 0, potn * sizeof(n));

			cudaMemcpy(dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy for device failed\n");

			dim3 fullBlocksPerGrid((potn + blockSize - 1) / blockSize);

			for (int k = 1; k < potn; k*=2)
			{
				upsweep <<< fullBlocksPerGrid, blockSize >>> (potn, k, dev);
			}

			cudaMemset(dev + potn - 1, 0, sizeof(int));

			for (int k = potn/2; k>0; k/=2)
			{
				downsweep <<< fullBlocksPerGrid, blockSize >>> (potn, k, dev);
			}

			cudaMemcpy(odata, dev, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy for output data failed\n");

			cudaFree(dev);
        }

	   __global__ void makeElementZero(int *data, int index) {
		   int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		   if (index == k) {
			   data[k] = 0;
		   }
	   }

	   __global__ void copyElements(int n, int *src, int *dest) {
		   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		   if (index >= n)
			   return;
		   dest[index] = src[index];
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
	   int compact(int n, PathSegment * dev_odata, PathSegment * dev_idata) {
		   int *dev_boolean;
		   int *dev_indices;
		   int count;

		   int paddedArraySize = 1 << ilog2ceil(n);

		   dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
		   dim3 fullBlocksPerGridPadded((paddedArraySize + blockSize - 1) / blockSize);

		   cudaMalloc((void**)&dev_boolean, paddedArraySize * sizeof(int));
		   checkCUDAError("Cannot allocate memory for boolean");
		   cudaMalloc((void**)&dev_indices, paddedArraySize * sizeof(int));
		   checkCUDAError("Cannot allocate memory for dev_indices");

		   StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_boolean, dev_idata);

		   copyElements << <fullBlocksPerGrid, blockSize >> >(n, dev_boolean, dev_indices);


		   for (int d = 0; d < ilog2ceil(paddedArraySize); d++) {
			   upsweep << <fullBlocksPerGridPadded, blockSize >> >(paddedArraySize, 1 << d, dev_indices);
		   }

		   makeElementZero << <fullBlocksPerGridPadded, blockSize >> >(dev_indices, paddedArraySize - 1);

		   for (int d = ilog2ceil(paddedArraySize) - 1; d >= 0; d--) {
			   downsweep << <fullBlocksPerGridPadded, blockSize >> >(paddedArraySize, 1 << d, dev_indices);
		   }

		   StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_odata, dev_idata, dev_boolean, dev_indices);

		   cudaMemcpy(dev_idata, dev_odata, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		   cudaMemcpy(&count, dev_indices + paddedArraySize - 1, sizeof(int), cudaMemcpyDeviceToHost);

		   cudaFree(dev_boolean);
		   cudaFree(dev_indices);
		   return count;
	   }
    }
}
