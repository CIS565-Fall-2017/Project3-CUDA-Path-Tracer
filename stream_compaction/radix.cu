#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "sharedandbank.h"
#include "radix.h"

namespace StreamCompaction {
	namespace Radix {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		//Generates array e for radixSort
		//gen e	//for (int i = 0; i < n; ++i) { e[i] = (ibuf[i] & mask) == mask ? 0 : 1; }
		__global__ void kernGenE(const int n, const int mask, int* e, const int* ibuf) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			e[index] = (ibuf[index] & mask) == mask ? 0 : 1;
		}

		//Generates array t for radixSort
		//gen t //for (int i = 0; i < n; ++i) { t[i] = i - f[i] + totalFalses; }
	  	__global__ void kernGenT(const int n, int* t, const int* f, const int totalFalses) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			t[index] = index - f[index] + totalFalses;
		}

		//Generates array d for radixSort
		//gen d //for (int i = 0; i < n; ++i) { d[i] = e[i] == 0 ? t[i] : f[i]; }
		__global__ void kernGenD(const int n, int* d, const int* e, const int* t, const int* f) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			d[index] = e[index] == 0 ? t[index] : f[index];
		}

		//Generates scattered output based on array d for radixSort
		//gen Scatter based on d  //for (int i = 0; i < n; ++i) { odata[d[i]] = ibuf[i]; }
		__global__ void kernGenScatterBasedOnD(const int n, int* odata, const int* d, const int* ibuf) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			odata[d[index]] = ibuf[index];
		}

		void radixSort(const int n, const int numbits, int* odata) {
			if (n <= 1) return;

			int* dev_odata;
			int* dev_ibuf;
			int* dev_e;
			int* e = new int[n];
			int* dev_f;
			int* f = new int[n];
			int* dev_t;
			int* dev_d;
			const int numbytes_copy = n * sizeof(int);
			const int pow2roundedsize = 1 << ilog2ceil(n);
			const int numbytes_pow2roundedsize = pow2roundedsize * sizeof(int);

			cudaMalloc((void**)&dev_odata, numbytes_copy);
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_ibuf, numbytes_copy);
			checkCUDAError("cudaMalloc dev_ibuf failed!");

			cudaMalloc((void**)&dev_e, numbytes_copy);
			checkCUDAError("cudaMalloc dev_e failed!");

			cudaMalloc((void**)&dev_f, numbytes_pow2roundedsize);
			checkCUDAError("cudaMalloc dev_f failed!");

			cudaMalloc((void**)&dev_t, numbytes_copy);
			checkCUDAError("cudaMalloc dev_t failed!");

			cudaMalloc((void**)&dev_d, numbytes_copy);
			checkCUDAError("cudaMalloc dev_d failed!");

			cudaMemcpy(dev_odata, odata, numbytes_copy, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from odata to dev_odata failed!");

			const int gridDim = (numbytes_copy + blockSize - 1) / blockSize;

	        timer().startGpuTimer();
			for (int mask = 1; mask <= (1 << numbits); mask <<= 1) {
				std::swap(dev_odata, dev_ibuf);
				//gen e	//for (int i = 0; i < n; ++i) { e[i] = (ibuf[i] & mask) == mask ? 0 : 1; }
				kernGenE << <gridDim, blockSize >> > (n, mask, dev_e, dev_ibuf);

				//{//generate f (scan)
				//	cudaMemcpy(dev_f, dev_e, numbytes_copy, cudaMemcpyDeviceToDevice);
				//	checkCUDAError("cudaMemcpy from to dev_e dev_f failed!");

				//	int gridDim = (pow2roundedsize + blockSize - 1) / blockSize;

				//	//the algo works on pow2 sized arrays so we size up the array to the next pow 2 if it wasn't a pow of 2 to begin with
				//	//then we need to fill data after index n-1 with zeros 
				//	StreamCompaction::Efficient::kernZeroExcessLeaves<<<gridDim, blockSize>>>(pow2roundedsize, n, dev_f);

				//	for (int offset = 1; offset < pow2roundedsize; offset <<= 1) {
				//		gridDim = ((pow2roundedsize >> ilog2(offset<<1)) + blockSize - 1) / blockSize;
				//		StreamCompaction::Efficient::kernScanUp<<<gridDim, blockSize>>>(pow2roundedsize, offset << 1, offset, dev_f);
				//	}

				//	//make sure last index value is 0 before we downsweep
				//	const int zero = 0;
				//	cudaMemcpy(dev_f + pow2roundedsize - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
				//	checkCUDAError("cudaMemcpy from zero to dev_data failed!");

				//	for (int offset = pow2roundedsize >> 1; offset > 0; offset >>= 1) {
				//		gridDim = ((pow2roundedsize >> ilog2(offset<<1)) + blockSize - 1) / blockSize;
				//		StreamCompaction::Efficient::kernScanDown<<<gridDim, blockSize>>>(pow2roundedsize, offset << 1, offset, dev_f);
				//	}
				//}//end generate f (scan)

				cudaMemcpy(dev_f, dev_e, numbytes_copy, cudaMemcpyDeviceToDevice);
				checkCUDAError("cudaMemcpy from to dev_e dev_f failed!");
				StreamCompaction::SharedAndBank::scanNoMalloc(pow2roundedsize, dev_f);

				//find totalFalses
				int eEND, fEND;
				cudaMemcpy(&eEND, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAError("cudaMemcpy from dev_e + n -1 to eEND failed!");
				cudaMemcpy(&fEND, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAError("cudaMemcpy from dev_f + n -1 to fEND failed!");
				const int totalFalses = eEND + fEND;

				//gen t //for (int i = 0; i < n; ++i) { t[i] = i - f[i] + totalFalses; }
				kernGenT<<<gridDim, blockSize>>>(n, dev_t, dev_f, totalFalses);
				
				//gen d //for (int i = 0; i < n; ++i) { d[i] = e[i] == 0 ? t[i] : f[i]; }
				kernGenD << <gridDim, blockSize >> > (n, dev_d, dev_e, dev_t, dev_f);

				//gen Scatter based on d  //for (int i = 0; i < n; ++i) { odata[d[i]] = ibuf[i]; }
				kernGenScatterBasedOnD<<<gridDim, blockSize>>>(n, dev_odata, dev_d, dev_ibuf);
			}

	        timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, numbytes_copy, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata odata failed!");

			cudaFree(dev_odata);
			checkCUDAError("cudaFree dev_odata failed!");
			cudaFree(dev_ibuf);
			checkCUDAError("cudaFree dev_ibuf failed!");
			cudaFree(dev_e);
			checkCUDAError("cudaFree dev_e failed!");
			cudaFree(dev_f);
			checkCUDAError("cudaFree dev_f failed!");
			cudaFree(dev_t);
			checkCUDAError("cudaFree dev_t failed!");
			cudaFree(dev_d);
			checkCUDAError("cudaFree dev_d failed!");
		}
	}
}