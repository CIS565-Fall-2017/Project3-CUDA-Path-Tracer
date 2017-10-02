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
	int loadMesh();
	void loadTransformations(Geom *newGeom, glm::vec3 translate, glm::vec3 rotate, glm::vec3 scale);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
	std::vector<Geom> triangles;
    std::vector<Material> materials;
    RenderState state;
};
