#pragma once

#include "sampling.h"

__host__ __device__ Color3f f_Specular(const Material &m)
{
	return m.specular.color;
}

__host__ __device__ float pdf_Specular()
{
	return 0.0f;
}