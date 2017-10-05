#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"


namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kern_initializeHistogram(int n, int* hists) {
			int i = threadIdx.x + (blockIdx.x * blockDim.x);
			if (i >= n) { return; }

			hists[i] = 0;
		}


		__global__ void kern_generateHistogram(int n, int bit, int desired_bit, const int* idata, int* hists) {
			int i = threadIdx.x + (blockIdx.x * blockDim.x);
			if (i >= n) { return; }

			int nth_bit = (idata[i] >> bit) & 1;

			//Determine if Zero
			hists[i] = (int)(nth_bit == desired_bit);
		}

		__global__ void kern_placeItems(int n, int* odata, const int* idata, const int* hists0, const int* hists1,
			const int* offset0, const int* offset1) {
			int i = threadIdx.x + (blockIdx.x * blockDim.x);
			if (i >= n) { return; }

			int zero_count = offset0[n - 1] + hists0[n - 1];

			//Zero Bit
			if (hists0[i] == 1) {
				odata[offset0[i]]              = idata[i];
			} else {
				odata[zero_count + offset1[i]] = idata[i];
			}
		}

        /**
         * Performs radix sort on a buffer, returns sorted array in odata
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to sort the elements.
         * @param idata  The array of elements to sort.
         */
        void sort(int n, int *odata, const int *idata) {
			try { timer().startGpuTimer(); }
			catch (...) {};
			//Define Thread and Block Counts
			dim3 threadsPerBlock(std::min(getThreadsPerBlock(), n));
			dim3 numBlocks(std::ceilf(((float)n / threadsPerBlock.x)));

			//Allocate GPU Buffers
			int* idata_GPU, *odata_GPU;
			cudaMalloc((void**)&idata_GPU, sizeof(int) * n);
			cudaMalloc((void**)&odata_GPU, sizeof(int) * n);
			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int* emptyCounts = (int*) malloc(sizeof(int) * 2 * n);
			//Get Maximum Number
			int max = -INT_MAX;
			for (int i = 0; i < n; i++) {
				max = std::max(idata[i], max);
				
				//Zero out array of counts
				*(emptyCounts + 0 * n + i) = 0;
				*(emptyCounts + 1 * n + i) = 0;
			}
			int max_MSB = ilog2ceil(max);

			// ---- For each bit:
			for (int bit = 0; bit < max_MSB; bit++) {
				//Create a hist[n] Array, with each thread writing to its index
				int* histograms_zero; // This is an Array of counts of zero
				cudaMalloc((void**)&histograms_zero, sizeof(int) * n);

				//Create a hist[n] Array, with each thread writing to its index
				int* histograms_one; // This is an Array of counts of zero
				cudaMalloc((void**)&histograms_one, sizeof(int) * n);

				//Create a prefixSum array that lets you do cool stuff to it
				int* offset_zero;
				cudaMalloc((void**)&offset_zero, sizeof(int) * n);

				//Create a prefixSum array that lets you do cool stuff to it
				int* offset_one;
				cudaMalloc((void**)&offset_one, sizeof(int) * n);

				// kern: Generate Histogram with Number of zero's and one's for each bit
				kern_generateHistogram << <numBlocks, threadsPerBlock >> > (n, bit, 0, idata_GPU, histograms_zero);
				kern_generateHistogram << <numBlocks, threadsPerBlock >> > (n, bit, 1, idata_GPU, histograms_one);

				// kern: Calculate Offsets for each 
				StreamCompaction::Efficient::scan(n, offset_zero, histograms_zero);
				StreamCompaction::Efficient::scan(n, offset_one, histograms_one);

				// kern: Ex Prefix Sum on Histogram
				//REMEMBER THE MEMCPY OPTIMIZATION
				/**
				int last_zero_elt, last_offset_zero;
				cudaMemcpy(&last_zero_elt, &histograms_zero[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&last_offset_zero, &offset_zero[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
				int zero_count = last_offset_zero + last_zero_elt;
				**/

				// kern: Place each thing int its location
				kern_placeItems << <numBlocks, threadsPerBlock >> > (n, odata_GPU, idata_GPU, histograms_zero, histograms_one, 
					                                                                      offset_zero,     offset_one);

				// Ping Pong Buffers
				std::swap(idata_GPU, odata_GPU);

				//Free Old Stuff
				cudaFree(histograms_zero);
				cudaFree(histograms_one);
				cudaFree(offset_zero);
				cudaFree(offset_one);
			}

			cudaMemcpy(odata, idata_GPU, sizeof(int) * n, cudaMemcpyDeviceToHost);

			cudaFree(idata_GPU);
			cudaFree(odata_GPU);
			free(emptyCounts);

			try { timer().endGpuTimer(); }
			catch (...) {};
 
        }
    }
}
