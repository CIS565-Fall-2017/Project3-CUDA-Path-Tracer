#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

/**
* This class is used for performance timing
* Adapted from WindyDarian(https://github.com/WindyDarian)
*/

class PerformanceTimer
{
public:
	//------------------------------------------------
	//Timer Variables and Functions
	//------------------------------------------------
	using time_point_t = std::chrono::high_resolution_clock::time_point;
	bool cpuTimerStarted = false;
	time_point_t timeStartCpu;
	time_point_t timeEndCpu;
	float prevElapsedTime_CPU_milliseconds = 0.0f;

	void startCpuTimer()
	{
		if (cpuTimerStarted) { throw std::runtime_error("CPU timer already started"); }
		cpuTimerStarted = true;

		timeStartCpu = std::chrono::high_resolution_clock::now();
	}

	void endCpuTimer()
	{
		timeEndCpu = std::chrono::high_resolution_clock::now();

		if (!cpuTimerStarted) { throw std::runtime_error("CPU timer not started"); }

		std::chrono::duration<double, std::milli> duro = timeEndCpu - timeStartCpu;
		prevElapsedTime_CPU_milliseconds =
			static_cast<decltype(prevElapsedTime_CPU_milliseconds)>(duro.count());

		cpuTimerStarted = false;
	}

	void printTimerDetails(int depth, int iteration)
	{
		printf("Iteration %d; Depth %d: %f milliseconds\n", depth, iteration, prevElapsedTime_CPU_milliseconds);
	}
};