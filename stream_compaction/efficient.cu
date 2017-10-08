#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <vector>
// Shared memory is 50 Kbyte per SM and an int is 4 Bytes so 
// if the TILE is greater than MAXTILE the array will not fit
// in shared memory
static constexpr int MAXTILE_Shared { 8192};
static constexpr int MAXTILE { ( MAXTILE_Shared < 2 * StreamCompaction::Efficient::blockSize) ?
	MAXTILE_Shared : 2 * StreamCompaction::Efficient::blockSize};

//static constexpr int devStart{ 0 };
//static constexpr int scanStart{ 0 };
static constexpr bool printDebug{ false };

void printArray2(int n, int *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15  && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%3d ", a[i]);
	}
	printf("]\n");
}



namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		// initialize the arrays on the GPU
		// returns the addresses of the pointers on the GPU
		// dev_idata is a pointer to the address of the dev_idata array that
		// gets updated here
		// initialize dev_idata has the first
		// elements copied and the remainder to make the stream 2^n
		// are set to 0. The first input is the size of the arrays
		// to allocate and the second input is the size of the array to transfer.
		// N the maximum size of the allocated array.  n is the size of the data array
		// N is one more than the multiple of 2 greater or equal to n, 
		// in dev_idata, and then the elements are copied inte dev_idata.

		void initScan(int N, int n, const int *idata, int ** dev_idata)
		{
			int size{ sizeof(int) };
			cudaMalloc(reinterpret_cast<void**>(dev_idata), N * size);
			checkCUDAError("Allocating Scan Buffer Efficient Error");
			cudaMemcpy(static_cast<void*>(*dev_idata), idata, n *size, cudaMemcpyHostToDevice);
			cudaMemset(static_cast<void*>(*dev_idata + n), 0, (N - n) * size);
			// no need to initialize the odata because the loop does that each time
			checkCUDAError("Initialize and Copy data to target Error");
			cudaThreadSynchronize();
		}
		void initScanSum(int numblocks, int ** scan_sum)
		{
		        int size {sizeof(int)};
			cudaMalloc(reinterpret_cast<void**>(scan_sum), numblocks * size);
			checkCUDAError("Allocating Scan Efficient shared  Error");
		
		}

		// transfer scan data back to host
		void transferIntToHost(int N, int * odata, int * dev_odata)
		{
			cudaMemcpy(odata, dev_odata, N * sizeof(int), cudaMemcpyDeviceToHost);
		}
		void printDevArray(int n, int start, int* devA, bool abridged = false)
		{
			if (start >= n || !printDebug) {
				return;
			}
			int * copy = new int[n - start];
		        transferIntToHost(n - start, copy, devA + start);
			printf("sIdx %d: ", start);
			printArray2(n - start,  copy, abridged);
			delete[] copy;
		}
		// end the scan on the device.
		void endScan(int * dev_idata)
		{
			cudaFree(dev_idata);
		}
            void freeScanShared(const SharedScan scan){
			cudaFree(scan.dev_idata);
			cudaFree(scan.scan_sum);
		}
		void freeCompaction(const CompactSupport compactSupport)
		{
			freeScanShared(compactSupport.scan);
			cudaFree(compactSupport.bool_data);
		}
		// Kernel Reduction and downsweep in Shared Memory. TILE is the width of the TILE 
		// and the max thread is TILE/2.

		__global__ void kernScanShared(int TILE, int * dev_idata, int * dev_Tilesum)
		{
                   extern __shared__ int  xy[];	
		   // load data into shared memory
		   // each thread loads two data points;
		   int maxThreads = TILE >> 1;
		   if (threadIdx.x >= maxThreads) {
			   return;
		   }
		   // copy to shared memory 1 thread per two data points
		   int Stride {2};
		   int Sharedindex = threadIdx.x * Stride;
		   int devindex = Sharedindex + TILE * blockIdx.x;
		   xy[Sharedindex] = dev_idata[devindex];
		   xy[Sharedindex + 1] = dev_idata[devindex + 1];
		   __syncthreads();
		   
		   // do the parallel reduction
		   int maxRed{ maxThreads };
		   for ( ; Stride <= TILE; Stride <<= 1, maxRed >>= 1)
		   {

			  int priorStride{ Stride >> 1 };
			  if (threadIdx.x < maxRed) {
				  int rindex = (threadIdx.x + 1) * Stride - 1;
				  xy[rindex] += xy[rindex - priorStride];
			  }
			  __syncthreads();
			//if (rindex < TILE){}
		   }
		   
		   const int startOffset { TILE - 1};
		   // have one thread in the block copy the last element;
		   // to scan and set that element to zero
           if ( threadIdx.x == 0) {
			   *(dev_Tilesum + blockIdx.x) = xy[startOffset];
			   xy[startOffset] = 0;
		   }
           __syncthreads();
		   // Now do the Downsweep to Sum elements
		   // Stride starts at TILE
		  maxRed = 1;
		  for (Stride = TILE; Stride > 1; Stride >>= 1, maxRed <<= 1){
			  if (threadIdx.x < maxRed) {
				  int right = -Stride * threadIdx.x + startOffset;
				  int separation = Stride >> 1;
				  int left = right - separation;
				  int current = xy[right];
				  xy[right] += xy[left];
				  xy[left] = current;
			  }
			//if (right >= 0) {} 
			__syncthreads();
		   } 

                   // now copy back;
		   dev_idata[devindex] = xy[Sharedindex];
		   dev_idata[devindex + 1] = xy[Sharedindex + 1];
		   __syncthreads();
		}
	
        __global__ void kernAddSumToTile(int Tile, int* dev_idata, int* ScanSum)
		{
           extern __shared__ int xy[];	
		   int maxThreads = Tile >> 1;
		   if (threadIdx.x >= maxThreads) {
			   return;
		   }
		   int Stride {2};
		   int Sharedindex = threadIdx.x * Stride;
		   int devindex = Sharedindex + Tile * blockIdx.x;
		   xy[Sharedindex] = dev_idata[devindex];
		   xy[Sharedindex + 1] = dev_idata[devindex + 1];
		   int  sum { *(ScanSum + blockIdx.x)};
		   xy[Sharedindex] += sum;
		   xy[Sharedindex + 1] += sum;
                   // now copy back;
		   dev_idata[devindex] = xy[Sharedindex];
		   dev_idata[devindex + 1] = xy[Sharedindex + 1];
		}
		// kernParallelReduction uses contiguous threads to do the parallel reduction
		// There is one thread for every two elements
		__global__ void kernParallelReduction(int N, int Stride,
			int maxThreads, int * dev_idata)
		{
			int thread = threadIdx.x + blockIdx.x * blockDim.x;
			if (thread >= maxThreads) {
				return;
			}
			int priorStride{ Stride >> 1 };
			int index = (thread + 1) * Stride - 1;
			if (index < N) {
				dev_idata[index] += dev_idata[index - priorStride];
			}
		}
		// Downsweep uses contiguous threads to sweep down and add the intermediate
		// results to the partial sums already computed
		// There is one thread for every two elements.  Here there is a for loop 
		// that changes the stride.  Contiguous allows the first threads to do all
		// the work and later warps will all be 0.
		__global__ void kernDownSweep(int N, int stride, int maxThreads, int * dev_idata)
		{
			int thread = threadIdx.x + blockIdx.x * blockDim.x;
			if (thread >= maxThreads) {
				return;
			}
			// have one thread set the last element to 0;
			int startOffset{ N - 1 };
			int right = -stride * thread + startOffset;
			if (right >= 0) {
				int separation = stride >> 1;
				int left = right - separation;
				int current = dev_idata[right];
				dev_idata[right] += dev_idata[left];
				dev_idata[left] = current;
			}
		}
		inline int gridSize(int threads) {
			return (threads + blockSize - 1) / blockSize;
		}
		/* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void efficientScan(int N, int d, int * dev_idata)
		{
			int maxThreads{ N >> 1 };
			for (int stride = { 2 }; stride <= N; stride *= 2)
			{
				int  grids{ gridSize(maxThreads) };
				dim3  fullBlocksPerGrid(grids);
				kernParallelReduction << <fullBlocksPerGrid, blockSize >> >
					(N, stride, maxThreads, dev_idata);
				maxThreads >>= 1;
			}
			cudaMemset((dev_idata + N - 1), 0, sizeof(int));
			maxThreads = 1;
			for (int stride = { N }; stride > 1; stride >>= 1)
			{
				    int grids{ gridSize(maxThreads) };
				    dim3  fullBlocksPerGrid(grids);
				//	printf(" %d %d %d\n", grids, maxThreads, stride);
				    kernDownSweep << <fullBlocksPerGrid, blockSize >> >(N, stride,
					maxThreads, dev_idata);
				   maxThreads <<= 1;
			}
			cudaThreadSynchronize();
		}
		// blocksNeeded calculates the number of scan blocks needed and 
		// the Tile Size.  N -- length of the original list that needs to
		// be a power of 2.  Tile is the Tile size or block size that also is a 
		// power of 2. dtile is Tile = 2^dtile.  The Tile size will change 
		// only if it is Tile > N. It produces 1 if N = 2^4 and T = 2^4 or if 
		// N = 2^3 and T = 2^4; log2N will be greater than 0. 
		// updates log2N and log2Tile.
		int blocksNeeded(int& log2N, int& log2Tile)
		{
			log2N -= log2Tile;
			if (log2N >= 0) {
				return 1 << log2N;
			}
			else {// set log2Tile to be the original log2N
				log2Tile += log2N;
				return 1;
			}
		}
	//	int blocksNeeded(int N, int& Tile, int dtile){
	//		if ( Tile > N) {
	//			Tile = N;
	//			return 1;
	//		}
	//		return N >> dtile;
	//	}
		// Totalblocks Needed  calculates the total tiles needed for 
		// all the scan arrays; log2N is log2(N) log2Tile is log2(Tile);
		int TotalBlocksNeeded(int log2N,  int log2Tile)
		{
			int total{0};
			log2N -= log2Tile;
			for (; log2N >= 0; log2N -= log2Tile)
			{
				total += 1 << log2N;
			}
			if (log2N > -log2Tile) {
				total += 1;
			}
			return total;
		}
		// this should also produce the total based on
		// the sum of a power series in 2^(log2Tile)
		int TotalBlocksNeeded2(int log2N, int log2Tile)
		{
			int quotient {log2N/log2Tile};
			int PowerofTile = quotient * log2Tile;
			int rem      = log2N - PowerofTile;
			int Total { ((1 << PowerofTile) - 1) / 
				    ((1 << log2Tile) -1)};
			if (rem != 0) {
				Total *= 1 << rem;
				Total += 1;
			}
			return Total;
		}
		// will call itself recursively to produce the Total Sum of idata
		// Tile is a multiple of 2 N is the size of dev_idata,
		// 2^(dtile) = Tile,  scan_sum is a preallocated array of the correct blocksize
		void efficientScanShared(int log2N, int log2Tile, int * dev_idata,  int* scan_sum, 
			                     int printoffset)
		{
			int n = (1 << log2N);
			// updates log2N for next iteration and log2Tile in case
			// this sum is less than the max
			printDevArray(n, printoffset, dev_idata, true);
			int numblocks {blocksNeeded(log2N, log2Tile)};
			int maxThreads{ 1 << (log2Tile - 1) };
			//maxThreads = std::max(maxThreads, 32);
			int Tile{ 1 << log2Tile };
			dim3 numThreads ( maxThreads);
			dim3 numBLOCKS(numblocks);
			int size { sizeof(int)};
			kernScanShared<<< numBLOCKS, numThreads, Tile * size >>>(
						Tile, dev_idata, scan_sum);
			checkCUDAError("find Cuda Error");
			printDevArray(n, printoffset, dev_idata, true);
			// no need for scan_sum and total scan is correct if numblocks == 1
			if (numblocks == 1) {
				return;
               		}
                        efficientScanShared( log2N,  log2Tile, scan_sum, 
					       scan_sum + numblocks, 0);
			numBLOCKS = dim3(numblocks - 1);
			kernAddSumToTile<<< numBLOCKS, numThreads, Tile * size >>>(
						Tile, dev_idata + Tile, scan_sum + 1);
			checkCUDAError("find Cuda Error 2");
			printDevArray(n, printoffset, dev_idata, true);
			//cudaThreadSynchronize();
		}

		// init device arrays necessary to sum N items in shared Memory
		SharedScan  initSharedScan(const int n, int tileSize)
		{
			// d is the number of scans needed and also the
			// upper bound for log2 of the number of elements
			SharedScan shared;
			shared.log2N = ilog2ceil(n); //
			int N{ 1 << shared.log2N };
			// Tile should be less than or equal to N, TILEMAX, Tile
			// inputted
			tileSize = (tileSize == -1) ? std::min(MAXTILE, N):
				              std::min({MAXTILE, N, tileSize});
			tileSize = std::max(tileSize, 2);
			shared.log2T = ilog2(tileSize);
			// Tile must be a multiple of 2 to divide N so Tile
			// will actually be less than this
			// calculate Total scan size
			int ScanSize { TotalBlocksNeeded(shared.log2N, shared.log2T)};
			//if (ScanSize != TotalBlocksNeeded2(shared.log2N, shared.log2T)){
			//	throw std::runtime_error("blocks needed may not be correct");
			//} can check if TotalBlocksNeeded is correct. It passed each time
			initScanSum(ScanSize, &shared.scan_sum);
			initScanSum(N, &shared.dev_idata);
			return shared;
		}
		CompactSupport initCompactSupport(const SharedScan scan){
			CompactSupport compact { scan, NULL};
			int N {1 << compact.scan.log2N};
			initScanSum(N, &compact.bool_data);
			return compact;
		}

	//  does the scan but now puts 
		void scanShared (int n, int *odata, const int *idata, int Tile)
		{
			int * dev_idata;
			int * scan_sum;
			// d is the number of scans needed and also the
			// upper bound for log2 of the number of elements
			int d{ ilog2ceil(n) }; //
			int N{ 1 << d };
			// Tile should be less than or equal to N, TILEMAX, Tile
			// inputted
			Tile = (Tile == -1) ? std::min(MAXTILE, N):
				              std::min({MAXTILE, N, Tile});
			Tile = std::max(Tile, 2);
			int dtile { ilog2(Tile)};
			// Tile must be a multiple of 2 to divide N so Tile
			// will actually be less than this
			// calculate Total scan size
			int ScanSize { TotalBlocksNeeded(d, dtile)};
			//if (ScanSize != TotalBlocksNeeded2(d, dtile)){
			//	throw std::runtime_error("blocks needed may not be correct");
			//}
			initScan(N, n, idata, &dev_idata);
			timer().startGpuTimer();
			initScanSum(ScanSize, &scan_sum);
			efficientScanShared(d, dtile, dev_idata, scan_sum);
			timer().endGpuTimer();
			// only transfer tho first n elements of the 
			// exclusive scan
			transferIntToHost(n, odata, dev_idata);
			endScan(dev_idata);
			endScan(scan_sum);
		}
		void scan(int n, int *odata, const int *idata)
		{
			int * dev_idata;
			// d is the number of scans needed and also the
			// upper bound for log2 of the number of elements
			int d{ ilog2ceil(n) }; //
			int N{ 1 << d };
			initScan(N, n, idata, &dev_idata);
			timer().startGpuTimer();
			efficientScan(N, d, dev_idata);
			timer().endGpuTimer();
			// only transfer tho first n elements of the 
			// exclusive scan
			transferIntToHost(n, odata, dev_idata);
			endScan(dev_idata);
		}

		void initCompact(int N, int n, const int *idata, int ** dev_idata,
			int **dev_booldata, int  ** dev_indices, int **dev_odata)
		{
			int size{ sizeof(int) };
			cudaMalloc(reinterpret_cast<void**> (dev_booldata), N * size);
			cudaMalloc(reinterpret_cast<void**> (dev_idata), N * size);
			cudaMalloc(reinterpret_cast<void**> (dev_indices), N * size);
			cudaMalloc(reinterpret_cast<void**> (dev_odata), N * size);
			checkCUDAError("Allocating Compaction Scan Error");
			cudaMemset(*dev_idata, 0, N * size);
			cudaMemcpy(*dev_idata, idata, n *size, cudaMemcpyHostToDevice);
			// no need to initialize the odata because the loop does that each time
			checkCUDAError("Initialize and Copy data to target Error");
			cudaThreadSynchronize();
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

			int * dev_idata;
			int * dev_booldata;
			int * dev_indices;
			int * dev_odata;
			// d is the number of scans needed and also the
			// upper bound for log2 of the number of elements
			int d{ ilog2ceil(n) }; //
			int N{ 1 << d };
			initCompact(N, n, idata, &dev_idata, &dev_booldata, &dev_indices,
				&dev_odata);
			timer().startGpuTimer();
			dim3  fullBlocksPerGrid((N + blockSize - 1) / blockSize);

			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid,
				blockSize >> >(N, dev_booldata, dev_idata);
			cudaMemcpy(dev_indices, dev_booldata, N * sizeof(int),
				cudaMemcpyDeviceToDevice);
			efficientScan(N, d, dev_indices);
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid,
				blockSize >> >(N, dev_odata, dev_idata,
					dev_booldata, dev_indices);
			timer().endGpuTimer();
			int  lastIndex;
			transferIntToHost(1, &lastIndex, dev_indices + N - 1);
			int lastIncluded;
			transferIntToHost(1, &lastIncluded, dev_booldata + N - 1);
			std::vector<int> input(n);
			std::vector<int> bools(n);
			std::vector<int> indices(n);
			transferIntToHost(n, input.data(), dev_idata);
			transferIntToHost(n, bools.data(), dev_booldata);
			transferIntToHost(n, indices.data(), dev_indices);
			printArray2(n,  input.data(), true);
			printArray2(n, bools.data(), true);
			printArray2(n, indices.data(), true);
			n = lastIncluded + lastIndex;
			transferIntToHost(n, odata, dev_odata);
			printArray2(n, odata, true);
			endScan(dev_odata);
			endScan(dev_idata);
			endScan(dev_indices);
			endScan(dev_booldata);
			return n;
		}

	}
}
