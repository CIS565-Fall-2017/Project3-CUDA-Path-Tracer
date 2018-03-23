#pragma once
#include "mesh.h"
class Shader;

//std
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

//assimp
enum aiTextureType;
class aiNode;
class aiMaterial;
struct aiScene;
struct aiMesh;

class Model {
public://data
	std::vector<Texture> mTexturesLoaded;//models will likely use textures more that once, only load it once
	std::vector<Mesh> mMeshes;
	std::string mDirectory;
	bool mirrored;
	mat4 modelmat;

public://functions
	Model(const std::string& path, const bool mirrored, const float unifscale);

	//loop through meshes in model and call mesh's draw function
	//void Draw(Shader shader);

	//load model using open asset importer (assimp)
	void loadModel(const std::string& path);

	//process the node in the scene, adding to mMeshes (call process mesh for each mesh in node)
	void processNode(aiNode* node, const aiScene const* scene);

	//process the mesh, handling mirrored uv's (eric lengyels book) and read textures from mesh store opengl id's int the mesh texture struct
	Mesh processMesh(const aiMesh* mesh, const aiScene* scene);

	//see if there's any tetuxtures of the specified type for the mesh material (calls utilit function that uploads texture to opengl and returns id using SOIL)
	std::vector<uint32_t> loadMaterialTextures(aiMaterial* mat, aiTextureType type);
};

