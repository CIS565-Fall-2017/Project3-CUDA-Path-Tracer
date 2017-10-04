#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <cuda.h>
#include <cuda_texture_types.h>
#include <texture_types.h>

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
	void loadOBJ(Geom& mesh, string& filename);
    int loadCamera();
	int loadEnvironment();

public:
	int light_count = 0;
	unsigned char* environment  = NULL;
	glm::ivec3 environment_dims = glm::ivec3(NULL);

    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
