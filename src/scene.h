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

	void loadObj(string objPath, Geom& newGeom, const glm::mat4& transform, const glm::mat4& invTranspose);

public:
    Scene(string filename);
    ~Scene();

	std::vector<Triangle> tris;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
