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
	int loadMesh(std::string Path, Geom &geom);
	std::string dir_path;
public:
    Scene(string filename);
    ~Scene();

	std::vector<unsigned int> lights_indices;
    std::vector<Geom> geoms;
	std::vector<Vertex> vertices;
    std::vector<Material> materials;
    RenderState state;

	int Smooth_Normals();
};
