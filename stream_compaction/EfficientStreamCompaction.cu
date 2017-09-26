#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "EfficientStreamCompaction.h"

namespace StreamCompaction {
    namespace Efficient {

		__global__ void kernSetZero(int N, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			dev_data[index] = 0;
		}

		__global__ void kernMapToBoolean(int N, int *bools, const PathSegment *idata) {
			// TODO
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			bools[index] = idata[index].remainingBounces ? 1 : 0;
		}

		//__global__ void kernScatter(int n, PathSegment *odata,
		//	PathSegment *idata, const int *bools, const int *indices) {
		//	int index = threadIdx.x + (blockIdx.x * blockDim.x);
		//	if (index >= n) {
		//		return;
		//	}

		//	if (bools[index]) {
		//		odata[indices[index]] = idata[index];
		//	}
		//}

		__global__ void kernSetCompactCount(int N, int* dev_count, int* bools, int* indices) {
			dev_count[0] = bools[N - 1] ? (indices[N - 1] + 1) : indices[N - 1];
		}


		__global__ void kernScanDynamicShared(int n, int *g_odata, int *g_idata, int *OriRoot) {
			extern __shared__ int temp[];

			int thid = threadIdx.x;
			// assume it's always a 1D block
			int blockOffset = 2 * blockDim.x * blockIdx.x;
			int offset = 1;

			temp[2 * thid] = g_idata[blockOffset + 2 * thid];
			temp[2 * thid + 1] = g_idata[blockOffset + 2 * thid + 1];

			// Up-sweep
			for (int d = n >> 1; d > 0; d >>= 1) {
				__syncthreads();
				if (thid < d) {
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;
					temp[bi] += temp[ai];
				}
				offset *= 2;
			}

			__syncthreads();
			// save origin root and set it to zero
			if (thid == 0) { 
				OriRoot[blockIdx.x] = temp[n - 1];
				temp[n - 1] = 0;
			}

			// Down-sweep
			for (int d = 1; d < n; d *= 2) {
				offset >>= 1;
				__syncthreads();
				if (thid < d) {
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;

					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			__syncthreads();
			g_odata[blockOffset + 2 * thid] = temp[2 * thid];
			g_odata[blockOffset + 2 * thid + 1] = temp[2 * thid + 1];
		}

		__global__ void kernAddOriRoot(int N, int stride, int* OriRoot, int* dev_odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			dev_odata[index] += OriRoot[blockIdx.x / stride];
		}

		void scanDynamicShared(int n, int *odata, const int *idata) {
			int* dev_data;

			dim3 blockDim(blockSize);
			dim3 gridDim((n + blockSize - 1) / blockSize);

			int size = gridDim.x * blockSize;

			cudaMalloc((void**)&dev_data, sizeof(int) * size);
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaDeviceSynchronize();

			int* ori_root;
			// ori_root_size has to be like that(first divide blockSize then multiply), 
			// because also needs to meet efficient algorithm requirement
			// eg. 
			// blockSize == 4,
			// indicies : 0 1 2 3 | 4 5 -> 0 1 2 3 | 4 5 0 0
			// elcusive_scan result : 0 0 1 3 | 0 4 9 9
			// ori_root : 6 9 (0 0)
			int ori_root_size = (gridDim.x + blockSize - 1) / blockSize;
			ori_root_size *= blockSize;

			cudaMalloc((void**)&ori_root, sizeof(int) * ori_root_size);
			//checkCUDAError("cudaMalloc ori_root failed!");
			cudaDeviceSynchronize();

			kernSetZero << < dim3((ori_root_size + blockDim.x - 1) / blockDim.x), blockDim >> > (ori_root_size, ori_root);
			//checkCUDAError("kernSetZero failed!");

			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			//checkCUDAError("cudaMemcpy failed!");

			int sharedMemoryPerBlockInBytes = blockDim.x * sizeof(int);


			// Step 1 : do scan
			kernScanDynamicShared << <gridDim, dim3(blockDim.x / 2), sharedMemoryPerBlockInBytes >> > (blockDim.x, dev_data, dev_data, ori_root);


			// Step 2.5 : scans of scan
			// like ori_root_size,
			// ori_root_of_ori_root_size has to align with blockSize
			int *ori_root_of_ori_root;
			int ori_root_of_ori_root_size = ori_root_size / blockSize;
			ori_root_of_ori_root_size = (ori_root_of_ori_root_size + blockSize - 1) / blockSize;
			ori_root_of_ori_root_size *= blockSize;

			cudaMalloc((void**)&ori_root_of_ori_root, sizeof(int) * ori_root_of_ori_root_size);
			//checkCUDAError("cudaMalloc ori_root_of_ori_root failed!");
			int stride = 1;

			do {
				// do scan of scan of scan here
				kernSetZero << < dim3((ori_root_of_ori_root_size + blockDim.x - 1) / blockDim.x), blockDim >> > (ori_root_of_ori_root_size, ori_root_of_ori_root);
				//checkCUDAError("kernSetZero failed!");

				kernScanDynamicShared << < dim3(ori_root_size / blockSize), dim3(blockDim.x / 2), sharedMemoryPerBlockInBytes >> > (blockDim.x, ori_root, ori_root, ori_root_of_ori_root);
				//checkCUDAError("kernScanDynamicShared 2 failed!");

				kernAddOriRoot << <gridDim, blockDim >> > (size, stride, ori_root, dev_data);
				//checkCUDAError("kernAddOriRoot failed!");

				// exit here
				// we exit until there is only one block
				if (ori_root_size == blockSize) {
					break;
				}

				// reset ori_root and ori_root_of_ori_root infomation
				ori_root_size = ori_root_of_ori_root_size;

				int *temp = ori_root_of_ori_root;
				ori_root_of_ori_root = ori_root;
				ori_root = temp;

				ori_root_of_ori_root_size = ori_root_size / blockSize;
				ori_root_of_ori_root_size = (ori_root_of_ori_root_size + blockSize - 1) / blockSize;
				ori_root_of_ori_root_size *= blockSize;

				stride *= blockSize;

			} while (true);


			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaFree(dev_data);
			cudaFree(ori_root);
			cudaFree(ori_root_of_ori_root);
		}

		int compactDynamicShared(int n, PathSegment *dev_data) {
			// compact Set-up
			int* bools;
			int* indices;
			int* dev_count;
			int count;

			dim3 blockDim(blockSize);
			dim3 gridDim((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&bools, n * sizeof(int));
			//checkCUDAError("cudaMalloc bools failed!");
			cudaMalloc((void**)&dev_count, sizeof(int));
			//checkCUDAError("cudaMalloc dev_count failed!");
			cudaDeviceSynchronize();


			// Scan Set-up
			// gridDim.x : has to be 2 ^n, which is our efficient compaction algorithm requires
			// size : acutal size + filled 0s
			int size = gridDim.x * blockSize;

			int* ori_root;
			// ori_root_size has to be like that(first divide blockSize then multiply), 
			// because also needs to meet efficient algorithm requirement
			// eg. 
			// blockSize == 4,
			// indicies : 0 1 2 3 | 4 5 -> 0 1 2 3 | 4 5 0 0
			// elcusive_scan result : 0 0 1 3 | 0 4 9 9
			// ori_root : 6 9 (0 0)
			int ori_root_size = (gridDim.x + blockSize - 1) / blockSize;
			ori_root_size *= blockSize;


			cudaMalloc((void**)&indices, size * sizeof(int));
			//checkCUDAError("cudaMalloc indices failed!");
			cudaMalloc((void**)&ori_root, sizeof(int) * ori_root_size);
			//checkCUDAError("cudaMalloc ori_root failed!");
			cudaDeviceSynchronize();

			kernSetZero << < gridDim, blockDim >> > (size, indices);
			//checkCUDAError("kernSetZero failed!");
			kernSetZero << < dim3((ori_root_size + blockDim.x - 1) / blockDim.x), blockDim >> > (ori_root_size, ori_root);
			//checkCUDAError("kernSetZero failed!");

			int sharedMemoryPerBlockInBytes = blockDim.x * sizeof(int);

			// Step 1 : compute bools array
			kernMapToBoolean << <gridDim, blockDim >> > (n, bools, dev_data);
			//checkCUDAError("kernMapToBoolean failed!");

			// indeices# >= bools#
			cudaMemcpy(indices, bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			//checkCUDAError("cudaMemcpy failed!");

			// Step 2 : exclusive scan indices
			kernScanDynamicShared << <gridDim, dim3(blockDim.x / 2), sharedMemoryPerBlockInBytes >> > (blockDim.x, indices, indices, ori_root);
			//checkCUDAError("kernScanDynamicShared 1 failed!");

			// Step 2.5 : scans of scan
			// like ori_root_size,
			// ori_root_of_ori_root_size has to align with blockSize
			int *ori_root_of_ori_root;
			int ori_root_of_ori_root_size = ori_root_size / blockSize;
			ori_root_of_ori_root_size = (ori_root_of_ori_root_size + blockSize - 1) / blockSize;
			ori_root_of_ori_root_size *= blockSize; 

			cudaMalloc((void**)&ori_root_of_ori_root, sizeof(int) * ori_root_of_ori_root_size);
			//checkCUDAError("cudaMalloc ori_root_of_ori_root failed!");
			int stride = 1;

			do {
				// do scan of scan of scan here
				kernSetZero << < dim3((ori_root_of_ori_root_size + blockDim.x - 1) / blockDim.x), blockDim >> > (ori_root_of_ori_root_size, ori_root_of_ori_root);
				//checkCUDAError("kernSetZero failed!");

				kernScanDynamicShared << < dim3(ori_root_size / blockSize), dim3(blockDim.x / 2), sharedMemoryPerBlockInBytes >> > (blockDim.x, ori_root, ori_root, ori_root_of_ori_root);
				//checkCUDAError("kernScanDynamicShared 2 failed!");

				kernAddOriRoot << <gridDim, blockDim >> > (size, stride, ori_root, indices);
				//checkCUDAError("kernAddOriRoot failed!");

				// exit here
				// we exit until there is only one block
				if (ori_root_size == blockSize) {
					break;
				}

				// reset ori_root and ori_root_of_ori_root infomation
				ori_root_size = ori_root_of_ori_root_size;

				int *temp = ori_root_of_ori_root;
				ori_root_of_ori_root = ori_root;
				ori_root = temp;

				ori_root_of_ori_root_size = ori_root_size / blockSize;
				ori_root_of_ori_root_size = (ori_root_of_ori_root_size + blockSize - 1) / blockSize;
				ori_root_of_ori_root_size *= blockSize;

				stride *= blockSize;

			} while (true);


			// Step 3 : Sort (Scatter)
			kernSetCompactCount << <dim3(1), dim3(1) >> > (n, dev_count, bools, indices);
			//checkCUDAError("kernSetCompactCount failed!");

			// Since scatter just discard elements who doesn't meet our criterion(bools value = 0)
			// However, we don't want discard pathSegments whoes remaining bounce is 0, we still its color info
			// after this iteration ends.
			// So, instead of scattering, we just sort here.
			// bools value == 1 will put ahead, and 0 behind.

			/*kernScatter << <gridDim, blockDim >> > (n, dev_data, dev_data, bools, indices);
			checkCUDAError("kernScatter failed!");*/

			thrust::device_ptr<int> dev_thrust_keys(bools);
			thrust::device_ptr<PathSegment> dev_thrust_values(dev_data);
			thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + n, dev_thrust_values, thrust::greater<int>());


			cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAError("cudaMemcpy failed!");


			cudaFree(bools);
			cudaFree(dev_count);
			cudaFree(indices);
			cudaFree(ori_root);
			cudaFree(ori_root_of_ori_root);

			return count;
		}
    }
}
