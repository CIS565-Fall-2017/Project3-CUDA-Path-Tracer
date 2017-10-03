#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <chrono>

using time_point_t = std::chrono::high_resolution_clock::time_point;
bool cpuTimerStarted = false;

cudaEvent_t event_start = nullptr;
cudaEvent_t event_end = nullptr;

time_point_t time_start_cpu;
time_point_t time_end_cpu;

bool cpu_timer_started = false;
bool gpu_timer_started = false;

float prev_elapsed_time_cpu_milliseconds = 0.f;
float prev_elapsed_time_gpu_milliseconds = 0.f;

void startCpuTimer() {
	if (cpuTimerStarted) {
		throw std::runtime_error("CPU timer already started");
	}

	cpuTimerStarted = true;
	time_start_cpu = std::chrono::high_resolution_clock::now();
}

void endCpuTimer() {
	time_end_cpu = std::chrono::high_resolution_clock::now();

	if (!cpuTimerStarted) {
		throw std::runtime_error("CPU timer not started");
	}

	std::chrono::duration<double, std::milli> duration = time_end_cpu - time_start_cpu;
	prev_elapsed_time_cpu_milliseconds = static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duration.count());

	cpuTimerStarted = false;
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

void printTime() {
	printf("Time Elapsed: %f milliseconds\n", prev_elapsed_time_cpu_milliseconds);
}

void printGPUTime() {
	printf("Time Elapsed: %f milliseconds\n", prev_elapsed_time_gpu_milliseconds);
}