#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
	namespace CPU {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		/**
		 * CPU scan (prefix sum).
		 * For performance analysis, this is supposed to be a simple for loop.
		 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
		 */
		void scan(int n, int *odata, const int *idata) {
			timer().startCpuTimer();
			scanImplementation(n, odata, idata);
			timer().endCpuTimer();
		}

		// A scan implementation to avoid running the CPU timer twice.
		// This idea is from a post in the Google group.
		void scanImplementation(int n, int* odata, const int* idata) {
			int total = 0;
			for (int i = 0; i < n; ++i) {
				odata[i] = total;
				total += idata[i];
			}
		}

		/**
		 * CPU stream compaction without using the scan function.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithoutScan(int n, int *odata, const int *idata) {
			timer().startCpuTimer();
			int total = 0;
			for (int i = 0; i < n; ++i) {
				int element = idata[i];
				if (element != 0) {
					odata[total] = element;
					++total;
				}
			}
			timer().endCpuTimer();
			return total;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int *odata, const int *idata) {
			timer().startCpuTimer();
			/*	Map the input array to an array of 0s and 1s, scan it, and use 
				scatter to produce the output. You will need a CPU scatter 
				implementation for this (see slides or GPU Gems chapter for an 
				explanation). */
			int* inputMap = new int[n];
			for (int i = 0; i < n; ++i) {
				int element = idata[i];
				if (element == 0) {
					inputMap[i] = 0;
				} else {
					inputMap[i] = 1;
				}
			}

			// Scan it
			scanImplementation(n, odata, inputMap);

			// Scatter
			int total = 0;
			for (int i = 0; i < (n - 1); ++i) {
				if (inputMap[i] == 1) {
					odata[odata[i]] = idata[i];
					++total;
				}
			}
			timer().endCpuTimer();
			return total;
		}
	}
}
