#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <windows.h>

#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "Stringapiset.h"
#include "tiny_obj_loader.h"

//using namespace std;

char * ConvertWCtoC(const wchar_t* str);

bool Inbound(AABB big, AABB sMall);

bool bSpan(AABB big, AABB sMall);

static void createOCtree(std::vector<Octree> &octreeStructure, Octree &thisOctree, std::vector<Triangle> &triangles);


class Scene {
private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    int loadGeom(std::string objectid);
    int loadCamera();
	int loadTexture();
public:
    Scene(std::string filename);
	~Scene()
	{
		/*
		for (int i = 0; i < geoms.size(); i++)
		{
			if (geoms[i].type == MESH)
			{
				delete[] geoms[i].meshInfo.triangles;
			}

			
		}
		*/
	}

    std::vector<Geom> geoms;
	std::vector<Material> materials;
	std::vector<Geom> lights;

	std::vector<Triangle> trianglesInMesh;
	std::vector<Octree> octreeforMeshes;


	std::vector<KDtreeNode> kdTree;
	std::vector<KDtreeNodeForGPU> kdTreeForGPU;
	std::vector<int> kdTreeTriangleIndexForGPU;

	std::vector<Image> Images;
	std::vector<glm::vec3> imageData;

	int envMapId;

    RenderState state;
};

/*
Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2);
Bounds3f Union(const Bounds3f& b1, const Point3f& p);
Bounds3f Union(const Bounds3f& b1, const glm::vec4& p);
*/