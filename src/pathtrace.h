#pragma once

#include <vector>
//#include "scene.h"
#include "Octree.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
