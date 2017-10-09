#pragma once

#include "common.h"
//#include "../src/sceneStructs.h";
namespace StreamCompaction {
    namespace Efficient {

    constexpr int blockSize{ 64 };
	struct SharedScan {
		int log2N; //Log base2 of the maximum number of entries;
		int log2T; //log base2 of the Tile size;
		int *dev_idata; // the array to sum not inititialized
		int *scan_sum; // sum scan needed to get the sum of the indices
		              // allocated but not initialized
	};
	struct CompactSupport {
		SharedScan scan; //   all the data needed to carry out the Scan
		int *   bool_data; // the bool data that holds true or false
	};

        StreamCompaction::Common::PerformanceTimer& timer();
        // scanShared does the same scan but now it uses shared memory
        // and does the scan in blocks
        void scanShared(int n, int *odata, const int *idata, 
			int tileSize = -1);
        void scan(int n, int *odata, const int *idata);
	SharedScan initSharedScan(const int N, int tileSize = -1);
	
        // will call itself recursively to produce the Total Sum of idata
        // Tile is a multiple of 2 N is the size of dev_idata,
        // 2^(dtile) = Tile,  scan_sum is a preallocated array of the correct blocksize
	// all these fields are in SharedScan
        void efficientScanShared(int log2N, int log2Tile, int * dev_idata,  int* scan_sum, 
        	                     int printoffset = 0);
        CompactSupport initCompactSupport(const SharedScan scan);
	void freeCompaction( const CompactSupport compactSupport);
	void freeScanShared(const SharedScan scan);
        int compact(int n, int *odata, const int *idata);
        //  call MemCopy but transfer N integers from device_odata to odata on host
		void transferIntToHost(int N, int * odata, int * dev_odata);

    }
}
