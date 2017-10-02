#pragma once
#include "sceneStructs.h"

namespace StreamCompaction {
	namespace Radix {
		void RadixSort(int n, int *odata, int *idata);
		void RadixSort_Path_Interactions(int n, PathSegment *dev_paths, ShadeableIntersection *dev_interactions, int m_num);
	}
}