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
		__global__ void kern_scan(int n, int d, int* odata, const int* idata) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= n) { return; }

			int two_powd = 1 << (d - 1);
			//Using ternary as recommended in lecture
		    odata[k] = (k >= two_powd) ? idata[k - two_powd] + idata[k] : idata[k];
		}

		__global__ void kern_shiftRight(int n, int* odata, const int* idata) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= n) { return; }

			//Using ternary as recommended in lecture
			odata[k] = (k == 0) ? 0 : idata[k - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(getThreadsPerBlock(), n));
			dim3 numBlocks(std::ceilf(((float) n / threadsPerBlock.x)  ));

			//New GPU Buffers
			int* idata_GPU, *odata_GPU;
			cudaMalloc((void**)&idata_GPU, sizeof(int) * n);
			cudaMalloc((void**)&odata_GPU, sizeof(int) * n);
			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			for (int d = 1; d <= ilog2ceil(n); d++) {
				kern_scan <<<numBlocks, threadsPerBlock>>>(n, d, odata_GPU, idata_GPU);
				checkCUDAErrorFn("kern_scan failed with error");
				std::swap(idata_GPU, odata_GPU);
			}
			kern_shiftRight<< <numBlocks, threadsPerBlock >> > (n, odata_GPU, idata_GPU);

			cudaMemcpy(odata, odata_GPU, sizeof(int) * n, cudaMemcpyDeviceToHost);
			timer().endGpuTimer();

			cudaFree(idata_GPU);
			cudaFree(odata_GPU);

			//PRINTER
			/*
			printf("After: \n ( ");
			for (int i = 0; i < 15; i++) {
				printf("%d, ", odata[i]);
			}
			printf(" ... ) \n\n");
			*/
        }
    }
}
