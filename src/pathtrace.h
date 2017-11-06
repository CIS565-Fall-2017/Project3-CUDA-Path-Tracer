#pragma once

#include <vector>
#include "scene.h"
#include "stream_compaction\efficient.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
float pathtrace(uchar4 *pbo, int frame, int iteration);
