#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"

#include <chrono>
#include <ctime>


using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Scene* scene;
extern int iteration;

extern int width;
extern int height;

void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);


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

	float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
	{
		return prev_elapsed_time_cpu_milliseconds;
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

	float prev_elapsed_time_cpu_milliseconds = 0.f;
};

PerformanceTimer& timer();

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
	std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}