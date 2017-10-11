#pragma once

#include "src/sceneStructs.h"
#include <device_launch_parameters.h>

// To use efficient compaction algorithm
// blockSize == 2 ^ n
#define blockSize 64

inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

inline int ilog2ceil(int x) {
	return ilog2(x - 1) + 1;
}

namespace StreamCompaction {
    namespace Efficient {

		void scanDynamicShared(int n, int *odata, const int *idata);

		int compactDynamicShared(int n, PathSegment *dev_data);
    }
}
