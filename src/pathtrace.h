#pragma once

#include <vector>
#include "scene.h"
#include <stream_compaction\common.h>

namespace PathTracer {
	StreamCompaction::Common::PerformanceTimer& timer();
	void pathtraceInit(Scene *scene);
	void pathtraceFree();
	void pathtrace(uchar4 *pbo, int frame, int iteration, bool cacheFirstIntersect, bool depthOfField, float focalPoint, float lenseRadius, bool sort);
}

