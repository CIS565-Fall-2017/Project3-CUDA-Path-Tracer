#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blocksize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void kernScanHelper(int n, int pow2dMinus1, int *in, int *out)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			// Inclusive Scan
			if (index >= pow2dMinus1) {
				out[index] = in[index - pow2dMinus1] + in[index];
			}
			else {
				out[index] = in[index];
			}
		}

		__global__ void inclusiveToExclusive(int n, int *in, int *out)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			out[index] = index > 0 ? in[index - 1] : 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 fullBlocksPerGrid((n + blocksize - 1) / blocksize);

			int *in;
			int *out;
			cudaMalloc((void**)&in, n * sizeof(int));
			cudaMalloc((void**)&out, n * sizeof(int));
			cudaMemcpy(in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
            // TODO

			for (int d = 1; d <= ilog2ceil(n); d++) {
				int pow2dMinus1 = pow(2, d - 1);
				
				kernScanHelper << <fullBlocksPerGrid, blocksize >> > (n, pow2dMinus1, in, out);

				std::swap(in, out);
			}

			// Shift to the right
			inclusiveToExclusive << < fullBlocksPerGrid, blocksize >> > (n, in, out);
            
			timer().endGpuTimer();

			// Copy final values into odata
			cudaMemcpy(odata, out, sizeof(int) * n, cudaMemcpyDeviceToHost);

			cudaFree(in);
			cudaFree(out);
		}
    }
}
