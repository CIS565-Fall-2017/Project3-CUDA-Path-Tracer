#pragma once
#include <assimp/Importer.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

class Model;
class Shader;

struct Vertex {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec2 uv;
	glm::vec4 tan;
	glm::vec3 bitan;
	glm::vec3 col = glm::vec3(1.f, 1.f, 1.f);
};

struct Texture {
	unsigned int id;//gpu bound opaque ref
	uint8_t* data;
	std::string type;//diffuse, specular, normal, height, ..
	aiString path;//file path
	int width;
	int height;
	int channels;
	//~Texture() { free(data); }
};

class Mesh {
public: //Data
	std::vector<Vertex> mVertices;
	std::vector<uint32_t> mIndices;
	std::vector<uint32_t> mTextures;//these should be references to the model loaded textures to save memory for redundant textures in a model
	uint32_t mVAO;
	uint32_t texFlags;
	Model& parentModel;

public: //Functions
	Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<uint32_t> textures, uint32_t texFlags, Model& model);
	void Draw();

	//void Draw(Shader shader);
//private:
	//unsigned int mVBO, mIBO;//should VAO be private instead of public?

	//void setupMesh() {
	//	//tell gpu we want some buffers
	//	glGenVertexArrays(1, &mVAO);
	//	glGenBuffers(1, &mVBO);
	//	glGenBuffers(1, &mIBO);

	//	//"any subsequent VBO, EBO(IBO), attrib calls will be stored in VAO currently bound"
	//	glBindVertexArray(mVAO);

	//	//push all the vertex info
	//	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
	//	glBufferData(GL_ARRAY_BUFFER, mVertices.size() * sizeof(Vertex), &mVertices[0], GL_STATIC_DRAW);

	//	//push the vertex index info
	//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
	//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndices.size() * sizeof(unsigned int), &mIndices[0], GL_STATIC_DRAW);

	//	//tell the gpu how to access our vertex data and which shader locations those attributes will be mapped to
	//	unsigned int location = 0;//pos
	//	glEnableVertexAttribArray(location);
	//	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

	//	location = 1;//nor
	//	glEnableVertexAttribArray(location);
	//	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nor));

	//	location = 2;//uv
	//	glEnableVertexAttribArray(location);
	//	glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));

	//	location = 3;//tan
	//	glEnableVertexAttribArray(location);
	//	glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tan));

	//	location = 4;//bitan
	//	glEnableVertexAttribArray(location);
	//	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitan));

	//	//always good to practice to set everything back to defaults once configured
	//	glBindVertexArray(0);
	//}
};