#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 1024

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kern_upSweep(int n, int d, int* idata) {
			int index = (threadIdx.x + (blockIdx.x * blockDim.x));
			int k = index * (1 << d + 1);
			if (index >= n || k >= n) { return; }

			idata[k + (1 << d+1) - 1] += idata[k + (1 << d) - 1];
		}

		__global__ void kern_downSweep(int n, int d, int* idata) {
			int k = (threadIdx.x + (blockIdx.x * blockDim.x)) * (1 << d + 1);
			if (k >= n) { return; }

			int t = idata[k + (1 << d) - 1];
			idata[k + (1 << d)   - 1] = idata[k + (1 << d+1) - 1];
			idata[k + (1 << d+1) - 1] += t;
		}

		__global__ void roundN(int n, int nRounded, int* idataRounded, const int* idata) {
			int i = (threadIdx.x + (blockIdx.x * blockDim.x));
			if (i >= nRounded) { return; }

			idataRounded[i] = i >= n ? 0 : idata[i];
		}

		__global__ void kern_add_sums(int n, int* sums, int* data) {
			// The thread's ID within the entire grid
			int threadId = threadIdx.x + (blockIdx.x * blockDim.x);

			if (threadId >= n) { return; }

			int sumIdx = threadId / blockDim.x;

			data[threadId] += sums[sumIdx];
		}

		/**

		__global__ void kern_scan_shared_fixed(int* sums, int *odata, const int *idata) {
			//Copy the arary into shared memory
			extern __shared__ int temp[];

			//Halved because we're doing double the loading
			const int blockSizeHalf = BLOCK_SIZE * 0.5f;

			//The thread's ID within a block
			int threadId = threadIdx.x;
			int threadId2 = threadIdx.x + blockDim.x;

			// The thread's ID within the entire grid
			int threadId_global = threadIdx.x + (blockIdx.x * blockDim.x);
			int threadId2_global = threadId_global + blockDim.x;

			// Offsets for avoiding bank conflicts
			int offsetA = CONFLICT_FREE_OFFSET(threadId);
			int offsetB = CONFLICT_FREE_OFFSET(threadId2);

			// Load in elts
			temp[threadId  + offsetA] = idata[threadId_global];
			temp[threadId2 + offsetB] = idata[threadId2_global];

			// Offset
			int stride = 1;
		
			//Do the UpSweep
			//From d = n to 0, log_n times
            //#pragma unroll BLOCK_SIZE
			for (int d = blockSizeHalf >> 1; d > 0; d >>= 1) {
				__syncthreads();
				
				if (threadId < d)
				{
				  int ai = stride * (2*threadId + 1) - 1;
				  int bi = stride * (2*threadId + 2) - 1;
				
				  ai += CONFLICT_FREE_OFFSET(ai);
			      bi += CONFLICT_FREE_OFFSET(bi);

				  temp[bi] += temp[ai];
				}
				stride <<= 1;
			}
			
			int idxLast = blockSizeHalf - 1 + CONFLICT_FREE_OFFSET(blockSizeHalf - 1);

			//Set the Sum to the 
			if (sums != nullptr && threadId == 0) {
				sums[blockIdx.x] = temp[idxLast];
			}

			//Temp of n-1 = zero
			if (threadId == 0) { temp[idxLast] = 0; }
	
            //#pragma unroll BLOCK_SIZE
			for (int d = 1; d < blockSizeHalf; d <<= 1) {
				stride >>= 1;
				__syncthreads();

				if (threadId < d) {
					int ai = stride * (2* threadId + 1) - 1;
					int bi = stride * (2* threadId + 2) - 1;
					
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);

					int t = temp[ai];    //Save Left Child
					temp[ai] = temp[bi]; //Store Right Child in left Child's place
					temp[bi] += t;       //Right child += left child
				}
			}

			__syncthreads();
		

			//WRITE OUT DA VALUES
			odata[threadId_global] = temp[threadId + offsetA];
			odata[threadId_global] = temp[threadId + offsetB];
		}


	//	__global__ void kern_scan_shared(int n, int* sums, int *odata, const int *idata) {
	//		//Copy the arary into shared memory
	//		extern __shared__ int temp[];
	//
	//		//The thread's ID within a block
	//		int threadId = threadIdx.x;
	//
	//		// The thread's ID within the entire grid
	//		int threadId_global = threadIdx.x + (blockIdx.x * blockDim.x);
	//
	//		// Load in elts
	//		temp[threadId] = idata[threadId_global];
	//
	//		// Offset
	//		int stride = 1;
	//
	//		//Do the UpSweep
	//		//From d = n to 0, log_n times
	//		for (int d = n >> 1; d > 0; d >>= 1) {
	//			__syncthreads();
	//
	//			if (threadId < d)
	//			{
	//				int ai = stride * (2 * threadId + 1) - 1;
	//				int bi = stride * (2 * threadId + 2) - 1;
	//
	//				temp[bi] += temp[ai];
	//			}
	//			stride <<= 1;
	//		}
	//
	//		//Set the Sum to the 
	//		if (sums != nullptr && threadId == 0) {
	//			sums[blockIdx.x] = temp[n - 1];
	//		}
	//
	//		//Temp of n-1 = zero
	//		if (threadId == 0) { temp[n - 1] = 0; }
	//
	//		//Down Sweep
	//		for (int d = 1; d < n; d <<= 1) {
	//			stride >>= 1;
	//			__syncthreads();
	//
	//			if (threadId < d) {
	//				int ai = stride * (2 * threadId + 1) - 1;
	//				int bi = stride * (2 * threadId + 2) - 1;
	//
	//				int t = temp[ai];    //Save Left Child
	//				temp[ai] = temp[bi]; //Store Right Child in left Child's place
	//				temp[bi] += t;       //Right child += left child
	//			}
	//		}
	//
	//		__syncthreads();
	//
	//
	//		//WRITE OUT DA VALUES
	//		odata[threadId_global] = temp[threadId];
	//	}
	
		__global__ void kern_scan_shared(int n, int* sums, int *odata, const int *idata) {
			//Copy the arary into shared memory
			extern __shared__ int temp[];
	
			//The thread's ID within a block
			int threadId = threadIdx.x;
			int threadId2 = threadIdx.x + (n/2);
	
			// The thread's ID within the entire grid
			int threadId_global = threadIdx.x + (blockIdx.x * blockDim.x);
			int threadId2_global = threadId_global + (n/2);
	
			// Offsets for avoiding bank conflicts
			int offsetA = CONFLICT_FREE_OFFSET(threadId);
			int offsetB = CONFLICT_FREE_OFFSET(threadId2);
	
			// Load in elts
			temp[threadId  + offsetA] = idata[threadId_global];
			temp[threadId2 + offsetB] = idata[threadId2_global];
	
			// Offset
			int stride = 1;
	
			//Do the UpSweep
			//From d = n to 0, log_n times
			//#pragma unroll BLOCK_SIZE
			for (int d = n >> 1; d > 0; d >>= 1) {
				__syncthreads();
	
				if (threadId < d)
				{
					int ai = stride * (2 * threadId + 1) - 1;
					int bi = stride * (2 * threadId + 2) - 1;
	
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);
	
					temp[bi] += temp[ai];
				}
				stride <<= 1;
			}
	
			int idxLast = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
	
			//Set the Sum to the 
			if (sums != nullptr && threadId == 0) {
				sums[blockIdx.x] = temp[idxLast];
			}
	
			//Temp of n-1 = zero
			if (threadId == 0) { temp[idxLast] = 0; }
	
			//#pragma unroll BLOCK_SIZE
			for (int d = 1; d < n; d <<= 1) {
				stride >>= 1;
				__syncthreads();
	
				if (threadId < d) {
					int ai = stride * (2 * threadId + 1) - 1;
					int bi = stride * (2 * threadId + 2) - 1;
	
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);
	
					int t = temp[ai];    //Save Left Child
					temp[ai] = temp[bi]; //Store Right Child in left Child's place
					temp[bi] += t;       //Right child += left child
				}
			}
	
			__syncthreads();
	
	
			//WRITE OUT DA VALUES
			odata[threadId_global] = temp[threadId + offsetA];
			odata[threadId_global] = temp[threadId + offsetB];
		}
	

		//*
		//Performs prefix-sum (aka scan) on idata, storing the result into odata.
		//
		// *** THIS IS AN EFFICIENT VERSION USING SHARED MEMORY
		
		void scan_shared(int n, int *odata, const int *idata) {
			//First things first, round n to the nearest power of two
			int loops = ilog2ceil(n);
			int nRounded = 1 << loops;

			// Also round getThreadsPerBlock to nearest power of two to avoid AIOOB errors
			// An "if thread >= n" check would also work but that leads to more divergence.
			int threads = ilog2ceil(getThreadsPerBlock());
			int threadCount = 1 << threads;

			// Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(BLOCK_SIZE,nRounded));
			dim3 numBlocks(std::ceilf(((float) nRounded / threadsPerBlock.x)));

			int blockCount = numBlocks.x;

			printf("Rounding from: %d to %d because we have %d blocks that are %d ints big\n", n, nRounded, blockCount, threadsPerBlock.x);

			// A copy of idata on the GPU
			int *idata_GPU, *idataRounded_GPU, *odata_GPU, *isums_GPU, *osums_GPU;
			cudaMalloc((void**)&idata_GPU       , sizeof(int) * n);
			cudaMalloc((void**)&idataRounded_GPU, sizeof(int) * nRounded);
			cudaMalloc((void**)&odata_GPU       , sizeof(int) * nRounded);
			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			if (nRounded != n) {
				roundN << <numBlocks, threadsPerBlock >> > (n, nRounded, idataRounded_GPU, idata_GPU);
			}
			else {
				idataRounded_GPU = idata_GPU;
			}

			//printGPUArray(nRounded, idataRounded_GPU);

			try { timer().startGpuTimer(); }
			catch (...) {};

			//If all the data fits within one block
			if (blockCount == 1) {
				cudaMalloc((void**)&isums_GPU, sizeof(int) * blockCount);
				cudaMalloc((void**)&osums_GPU, sizeof(int) * blockCount);
				kern_scan_shared<<<numBlocks, threadsPerBlock, sizeof(int) * threadsPerBlock.x >> >(threadsPerBlock.x, nullptr, odata_GPU, idataRounded_GPU);
			}
			else { 
				//If we need to use the sums
				//Scan all the blocks separately
				cudaMalloc((void**)&isums_GPU, sizeof(int) * blockCount * 2.f);
				cudaMalloc((void**)&osums_GPU, sizeof(int) * blockCount * 2.f);
				blockCount *= 2;
				dim3 numBlocksDouble = dim3(blockCount);
				dim3 threadsPerBlockHalf = dim3(threadsPerBlock.x * 0.5f);
				kern_scan_shared_fixed << < numBlocksDouble, threadsPerBlockHalf , sizeof(int) * threadsPerBlock.x>> > (isums_GPU, odata_GPU, idataRounded_GPU);

				//printGPUArray(n, odata_GPU);
				//Run scan on the sums array
				kern_scan_shared <<<numBlocksDouble, blockCount, sizeof(int) * blockCount >>>       (blockCount, nullptr, osums_GPU, isums_GPU);
				//printGPUArray(blockCount, osums_GPU);

				//Add result of scan back to the per-block scans
				kern_add_sums << <numBlocksDouble, threadsPerBlockHalf >> > (n, osums_GPU, odata_GPU);
			}
			try { timer().endGpuTimer(); }
			catch (...) {};

			//printGPUArray(n, odata_GPU);

			//printf("\n PRINT THIS BITCH \n\n");
			//printGPUArray(blockCount, osums_GPU);

			cudaMemcpy(odata, odata_GPU, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(idata_GPU);
			cudaFree(idataRounded_GPU);
			cudaFree(odata_GPU);
			cudaFree(isums_GPU);
			cudaFree(osums_GPU);
		}

	*/
		
		__global__ void kern_scan_shared_fixed(int* sums, int *odata, const int *idata) {
			//Copy the arary into shared memory
			extern __shared__ int temp[];
		
			//The thread's ID within a block
			int threadId = threadIdx.x;
		
			// The thread's ID within the entire grid
			int threadId_global = threadIdx.x + (blockIdx.x * blockDim.x);
		
			// Offsets for avoiding bank conflicts
			//int offsetA = ;
			//int offsetB = ;
		
			// Load in elts
			temp[threadId] = idata[threadId_global];
		
			// Offset
			int stride = 1;
		
			//Do the UpSweep
			//From d = n to 0, log_n times
		    #pragma unroll BLOCK_SIZE
			for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
				__syncthreads();
		
				if (threadId < d)
				{
					int ai = stride * (2 * threadId + 1) - 1;
					int bi = stride * (2 * threadId + 2) - 1;
		
					temp[bi] += temp[ai];
				}
				stride <<= 1;
			}
		
			//Set the Sum to the 
			if (sums != nullptr && threadId == 0) {
				sums[blockIdx.x] = temp[BLOCK_SIZE - 1];
			}
		
			//Temp of n-1 = zero
			if (threadId == 0) { temp[BLOCK_SIZE - 1] = 0; }
		
			int offset = 0;
		
            #pragma unroll BLOCK_SIZE
			for (int d = 1; d < BLOCK_SIZE; d <<= 1) {
				stride >>= 1;
				__syncthreads();
		
				if (threadId < d) {
					int ai = stride * (2 * threadId + 1) - 1;
					int bi = stride * (2 * threadId + 2) - 1;
		
					int t = temp[ai];    //Save Left Child
					temp[ai] = temp[bi]; //Store Right Child in left Child's place
					temp[bi] += t;       //Right child += left child
				}
			}
		
			__syncthreads();
		
			//WRITE OUT DA VALUES
			odata[threadId_global] = temp[threadId];
		}
		
		__global__ void kern_scan_shared(int n, int* sums, int *odata, const int *idata) {
			//Copy the arary into shared memory
			extern __shared__ int temp[];
		
			//The thread's ID within a block
			int threadId = threadIdx.x;
		
			// The thread's ID within the entire grid
			int threadId_global = threadIdx.x + (blockIdx.x * blockDim.x);
		
			// Offsets for avoiding bank conflicts
			//int offsetA = ;
			//int offsetB = ;
		
			// Load in elts
			temp[threadId] = idata[threadId_global];
		
			// Offset
			int stride = 1;
		
			//Do the UpSweep
			//From d = n to 0, log_n times
			for (int d = n >> 1; d > 0; d >>= 1) {
				__syncthreads();
		
				if (threadId < d)
				{
					int ai = stride * (2 * threadId + 1) - 1;
					int bi = stride * (2 * threadId + 2) - 1;
		
					temp[bi] += temp[ai];
				}
				stride <<= 1;
			}
		
			//Set the Sum to the 
			if (sums != nullptr && threadId == 0) {
				sums[blockIdx.x] = temp[n - 1];
			}
		
			//Temp of n-1 = zero
			if (threadId == 0) { temp[n - 1] = 0; }
		
			int offset = 0;
		
			//Down Sweep
			for (int d = 1; d < n; d <<= 1) {
				stride >>= 1;
				__syncthreads();
		
				if (threadId < d) {
					int ai = stride * (2 * threadId + 1) - 1;
					int bi = stride * (2 * threadId + 2) - 1;
		
					int t = temp[ai];    //Save Left Child
					temp[ai] = temp[bi]; //Store Right Child in left Child's place
					temp[bi] += t;       //Right child += left child
				}
			}
		
			__syncthreads();
		
		
			//WRITE OUT DA VALUES
			odata[threadId_global] = temp[threadId];
		}
		
		//*
		// Performs prefix-sum (aka scan) on idata, storing the result into odata.
		//
		//*** THIS IS AN EFFICIENT VERSION USING SHARED MEMORY
		//
		void scan_shared(int n, int *odata, const int *idata) {
			//First things first, round n to the nearest power of two
			int loops = ilog2ceil(n);
			int nRounded = 1 << loops;
		
			// Also round getThreadsPerBlock to nearest power of two to avoid AIOOB errors
			// An "if thread >= n" check would also work but that leads to more divergence.
			int threads = ilog2ceil(getThreadsPerBlock());
			int threadCount = 1 << threads;
		
			// Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(BLOCK_SIZE, nRounded));
			dim3 numBlocks(std::ceilf(((float)nRounded / threadsPerBlock.x)));
		
			int blockCount = numBlocks.x;
		
			printf("Rounding from: %d to %d because we have %d blocks that are %d ints big\n", n, nRounded, blockCount, threadsPerBlock.x);
		
			// A copy of idata on the GPU
			int *idata_GPU, *idataRounded_GPU, *odata_GPU, *isums_GPU, *osums_GPU;
			cudaMalloc((void**)&idata_GPU, sizeof(int) * n);
			cudaMalloc((void**)&idataRounded_GPU, sizeof(int) * nRounded);
			cudaMalloc((void**)&odata_GPU, sizeof(int) * nRounded);
			cudaMalloc((void**)&isums_GPU, sizeof(int) * blockCount);
			cudaMalloc((void**)&osums_GPU, sizeof(int) * blockCount);
			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			if (nRounded != n) {
				roundN << <numBlocks, threadsPerBlock >> > (n, nRounded, idataRounded_GPU, idata_GPU);
			}
			else {
				idataRounded_GPU = idata_GPU;
			}
		
			//printGPUArray(nRounded, idataRounded_GPU);
		
			try { timer().startGpuTimer(); }
			catch (...) {};
		
			//If all the data fits within one block
			if (blockCount == 1) {
				kern_scan_shared << <numBlocks, threadsPerBlock, sizeof(int) * threadsPerBlock.x >> >(threadsPerBlock.x, nullptr, odata_GPU, idataRounded_GPU);
			}
			else { //If we need to use the sums
				   //Scan all the blocks separately
				kern_scan_shared_fixed << <numBlocks, threadsPerBlock, sizeof(int) * threadsPerBlock.x >> >(isums_GPU, odata_GPU, idataRounded_GPU);
		
				//Run scan on the sums array
				kern_scan_shared << <numBlocks, blockCount, sizeof(int) * blockCount >> >       (blockCount, nullptr, osums_GPU, isums_GPU);
		
				//Add result of scan back to the per-block scans
				kern_add_sums << <numBlocks, threadsPerBlock >> > (n, osums_GPU, odata_GPU);
			}
			try { timer().endGpuTimer(); }
			catch (...) {};
		
			//printGPUArray(n, odata_GPU);
		
			//printf("\n PRINT THIS BITCH \n\n");
			//printGPUArray(blockCount, osums_GPU);
		
			cudaMemcpy(odata, odata_GPU, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(idata_GPU);
			cudaFree(idataRounded_GPU);
			cudaFree(odata_GPU);
			cudaFree(isums_GPU);
			cudaFree(osums_GPU);
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(getThreadsPerBlock(), n));
			dim3 numBlocks(std::ceilf(((float)n / threadsPerBlock.x)));

			//Round Up
			int loops = ilog2ceil(n);
			int nRounded = 1 << loops;

			// A copy of idata on the GPU
			int* idata_GPU;
			cudaMalloc((void**)&idata_GPU, sizeof(int) * n);
			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			try { timer().startGpuTimer(); }
			catch (...) {};

			//Rounded Version of GPU Copy
			int* idataRounded_GPU;
			cudaMalloc((void**)&idataRounded_GPU, sizeof(int) * nRounded);
			//Round the GPU Array
			roundN << <numBlocks, threadsPerBlock >> > (n, nRounded, idataRounded_GPU, idata_GPU);

			//Up-Sweep:
			for (int d = 0; d < loops; d++) {
				kern_upSweep<< <numBlocks, threadsPerBlock >> >(n, d, idataRounded_GPU);
				checkCUDAErrorFn("upSweep failed with error");
			}

			//Set Zero
			int zero = 0;
			cudaMemcpy(&idataRounded_GPU[nRounded - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Zero Copy failed with error");

			//Down-Sweep:
			for (int d = loops - 1; d >= 0; d--) {
				kern_downSweep <<<numBlocks, threadsPerBlock >> >(nRounded, d, idataRounded_GPU);
				checkCUDAErrorFn("downSweep failed with error");
			}

			cudaMemcpy(odata, idataRounded_GPU, sizeof(int) * n, cudaMemcpyDeviceToHost);

			//Free Malloc'd
			cudaFree(idataRounded_GPU);
			cudaFree(idata_GPU);
			/**** PRINTER ******
			printf("After DownSweep: \n (");
			for (int i = nRounded-10; i < nRounded -1; i++) {
				printf("%d = %d, ", i, odata[i]);
			}
			printf("%d = %d) \n\n", nRounded-1, odata[nRounded-1]);
			**/
			try { timer().endGpuTimer(); }
			catch (...) {};
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
			try { timer().startGpuTimer(); }
			catch (...) {};
			// Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(getThreadsPerBlock(), n));
			dim3 numBlocks(std::ceilf(((float)n / threadsPerBlock.x)));

            // Create Buffers
			int *temp, *scanned, *idata_GPU, *odata_GPU, *count_GPU;
			cudaMalloc((void**)&temp     , sizeof(int) * n);
			cudaMalloc((void**)&scanned  , sizeof(int) * n);
			cudaMalloc((void**)&idata_GPU, sizeof(int) * n);
			cudaMalloc((void**)&odata_GPU, sizeof(int) * n);

			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("idata memcpy failed with error");
			
			//Create Temp Array
			Common::kernMapToBoolean << <numBlocks, threadsPerBlock >> > (n, temp, idata_GPU);
			checkCUDAErrorFn("kern_boolify failed with error");

			//Scan
			scan(n, scanned, temp);

			//Temporarily store "scanned" into odata to get count
			cudaMemcpy(odata, scanned, sizeof(int) * n, cudaMemcpyDeviceToHost);
			int count = odata[n - 1] + (int)(idata[n - 1] != 0);
			
			//Compact
			Common::kernScatter << <numBlocks, threadsPerBlock >> >(n, odata_GPU, idata_GPU, temp, scanned);
			checkCUDAErrorFn("kern_compact failed with error");

			//Bring Back to CPU
			cudaMemcpy(odata,  odata_GPU, sizeof(int) * count, cudaMemcpyDeviceToHost);
			
			//Free Up All Malloc'd
			cudaFree(temp);
			cudaFree(scanned);
			cudaFree(idata_GPU);
			cudaFree(odata_GPU);


			try { timer().endGpuTimer(); }
			catch (...) {};
            return count;
        }
    }
}
