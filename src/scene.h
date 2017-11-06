#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};

// from tiny_obj example functions
static void CalcNormal(float N[3], float v0[3], float v1[3], float v2[3]) {
	float v10[3];
	v10[0] = v1[0] - v0[0];
	v10[1] = v1[1] - v0[1];
	v10[2] = v1[2] - v0[2];

	float v20[3];
	v20[0] = v2[0] - v0[0];
	v20[1] = v2[1] - v0[1];
	v20[2] = v2[2] - v0[2];

	N[0] = v20[1] * v10[2] - v20[2] * v10[1];
	N[1] = v20[2] * v10[0] - v20[0] * v10[2];
	N[2] = v20[0] * v10[1] - v20[1] * v10[0];

	float len2 = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
	if (len2 > 0.0f) {
		float len = sqrtf(len2);

		N[0] /= len;
		N[1] /= len;
	}
}
