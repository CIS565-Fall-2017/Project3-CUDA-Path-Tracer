#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

//All of the following calculations were done for my GTX 1050 and 
//they will probably not work with most other non 6.1CC GPU's

// "x" in my master equation
#define MEM_PER_THREAD 4

// "z" in my master equation
#define OPTIMAL_BLOCKS_PER_SM 32

// Inequality/RHS in my master equation
#define SHARED_MEM_MAX 96000

// A muting term that tones done the optimality
#define TONE_DOWN 0.74f

//The number of warps on this GPU
#define WARP_SIZE 32



/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int binary(int num)
{
	if (num == 0)
	{
		return 0;
	}
	else
	{
		return (num % 2) + 10 * binary(num / 2);
	}
}

inline void printCPUArrayb(int n, int* arr) {
	printf("\n( ");
	for (int i = 0; i < n - 1; i++) {
		printf("%d, ", binary(arr[i]));
	}
	printf("%d)", binary(arr[n - 1]));
}

inline void printGPUArrayb(int n, int* dev_arr) {
	int* cpu_arr = (int*)malloc(sizeof(int) * n);
	cudaMemcpy(cpu_arr, dev_arr, sizeof(int)* n, cudaMemcpyDeviceToHost);
	printCPUArrayb(n, cpu_arr);
	free(cpu_arr);
}


inline void printCPUArray(int n, int* arr) {
	printf("\n( ");
	for (int i = 0; i < n-1; i++) {
		printf("%d, ", arr[i]);
	}
	printf("%d)", arr[n-1]);
}

inline void printGPUArray(int n, int* dev_arr) {
	int* cpu_arr = (int*)malloc(sizeof(int) * n);
	cudaMemcpy(cpu_arr, dev_arr, sizeof(int)* n, cudaMemcpyDeviceToHost);
	printCPUArray(n, cpu_arr);
	free(cpu_arr);
}

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int getThreadsPerBlock() {
	//Get theoretical best y value you can based on GPU specs
	//TODO: Find a better way to get the specs
	int theoretical_y = SHARED_MEM_MAX * TONE_DOWN / (MEM_PER_THREAD * OPTIMAL_BLOCKS_PER_SM);

	//Find closest multiple of 32 to theoretical_y
	for (int i = 0; i < SHARED_MEM_MAX / 32; i++) {
		if (i * 32 >= theoretical_y) {
			return (i - 1) * 32;
		}
	}
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}

namespace StreamCompaction {
    namespace Common {
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices);

	    /**
	    * This class is used for timing the performance
	    * Uncopyable and unmovable
        *
        * Adapted from WindyDarian(https://github.com/WindyDarian)
	    */
	    class PerformanceTimer
	    {
	    public:
		    PerformanceTimer()
		    {
			    cudaEventCreate(&event_start);
			    cudaEventCreate(&event_end);
		    }

		    ~PerformanceTimer()
		    {
			    cudaEventDestroy(event_start);
			    cudaEventDestroy(event_end);
		    }

		    void startCpuTimer()
		    {
			    if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
			    cpu_timer_started = true;

			    time_start_cpu = std::chrono::high_resolution_clock::now();
		    }

		    void endCpuTimer()
		    {
			    time_end_cpu = std::chrono::high_resolution_clock::now();

			    if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

			    std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
			    prev_elapsed_time_cpu_milliseconds =
				    static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

			    cpu_timer_started = false;
		    }

		    void startGpuTimer()
		    {
			    if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
			    gpu_timer_started = true;

			    cudaEventRecord(event_start);
		    }

		    void endGpuTimer()
		    {
			    cudaEventRecord(event_end);
			    cudaEventSynchronize(event_end);

			    if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

			    cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
			    gpu_timer_started = false;
		    }

		    float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
		    {
			    return prev_elapsed_time_cpu_milliseconds;
		    }

		    float getGpuElapsedTimeForPreviousOperation() //noexcept
		    {
			    return prev_elapsed_time_gpu_milliseconds;
		    }

		    // remove copy and move functions
		    PerformanceTimer(const PerformanceTimer&) = delete;
		    PerformanceTimer(PerformanceTimer&&) = delete;
		    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
		    PerformanceTimer& operator=(PerformanceTimer&&) = delete;

	    private:
		    cudaEvent_t event_start = nullptr;
		    cudaEvent_t event_end = nullptr;

		    using time_point_t = std::chrono::high_resolution_clock::time_point;
		    time_point_t time_start_cpu;
		    time_point_t time_end_cpu;

		    bool cpu_timer_started = false;
		    bool gpu_timer_started = false;

		    float prev_elapsed_time_cpu_milliseconds = 0.f;
		    float prev_elapsed_time_gpu_milliseconds = 0.f;
	    };
    }
}
