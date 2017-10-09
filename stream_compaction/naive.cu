#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
/// threads per block
static const int blockSize{ 256 };
namespace StreamCompaction {
    namespace Naive {
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
	// initialize the dev_odata to 0.0; dev_idata has the first
        // elements copied and the remainder to make the stream 2^n
        // are set to 0. The first input is the size of the arrays
        // to allocate and the second input is the size of the array to transfer.
	// N the maximum size of the allocated array.  n is the size of the data array
	// N is one more than the multiple of 2 greater or equal to n, 
	// 0 is placed at the first element
	// in dev_idata, and then the elements are copied inte dev_idata.
	// dev_odata has its first element initialized to 0 too.
	void initScan(int N, int n, const int *idata, int ** dev_odata, int ** dev_idata)
        {
            int size {sizeof(int)};
            cudaMalloc(reinterpret_cast<void**> (dev_idata), N * size);
	        cudaMalloc(reinterpret_cast<void**> (dev_odata), N * size);
            checkCUDAError("Allocating Scan Buffer Error"); 
	        cudaMemset(*dev_idata, 0,  N * size);
	        cudaMemset(*dev_odata, 0,  N * size);
	        cudaMemcpy(*dev_idata + 1, idata, n *size, cudaMemcpyHostToDevice);
	    // no need to initialize the odata because the loop does that each time
	        checkCUDAError("Initialize and Copy data to target Error");
	        cudaThreadSynchronize();
	}
        // transfer scan data back to host
	void transferScan(int N, int * odata, int * dev_odata)
	{
		cudaMemcpy(odata, dev_odata, N * sizeof(int), cudaMemcpyDeviceToHost);
	}
		
	// end the scan on the device.
	void endScan(int * dev_odata, int * dev_idata)
	{
		cudaFree(dev_idata);
		cudaFree(dev_odata);
	}
              
        
	
	// TODO:
	__global__ void kernOneNaiveScan(int N, int pow2d_1, int * dev_odata, int * dev_idata)
	{
		int k = threadIdx.x + blockIdx.x * blockDim.x;
		if (k >= N || (k < pow2d_1 >> 1)) {
			return;
		}
		dev_odata[k] = dev_idata[k];
		if ( k >= pow2d_1) {
			dev_odata[k] += dev_idata[k - pow2d_1];
		}
	}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, int indx) 
	{
            int * dev_idata;
            int * dev_odata;
            // d is the number of scans needed and also the
            // upper bound for log2 of the number of elements
            int d {ilog2ceil(n)}; //
	    // add one so that the 0th element is 0 for the 
	    // eclusive Scan
            int N { 1 << d };
            initScan(N + 1 , n, idata, &dev_odata, &dev_idata);	
            timer().startGpuTimer();
            dim3  fullBlocksPerGrid((N + blockSize - 1)/ blockSize);
	  //  int final { 1 << d - 1};
            for (int pow2d_1 {1}; pow2d_1 < N; pow2d_1 *= 2) {
		    // copy all elements to dev_odata to save
	            kernOneNaiveScan<<<fullBlocksPerGrid, blockSize>>>
			      (N, pow2d_1, dev_odata + 1, dev_idata + 1);
         	    std::swap(dev_odata, dev_idata);
		    if (pow2d_1 == indx) break;
	    }
            timer().endGpuTimer();
            // only transfer tho first n elements of the 
            // exclusive scan
            transferScan(n, odata, dev_idata);
            endScan(dev_odata, dev_idata);

        }
    }
}
