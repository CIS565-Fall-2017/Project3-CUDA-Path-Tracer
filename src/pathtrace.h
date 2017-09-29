#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, bool cacheFirstIntersect, bool depthOfField, float focalPoint, float lenseRadius, bool sort);
