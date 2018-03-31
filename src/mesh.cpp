#pragma once
#include <assimp/Importer.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "mesh.h"
#include "model.h"
#include <iostream>
//#include "shader.h"



//Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<uint32_t> textures, uint32_t texFlags, Model& model)
//	:mVertices(vertices), mIndices(indices), mTextures(textures), texFlags(texFlags), parentModel(model)
//{
//	//setupMesh();
//}

Mesh::Mesh( uint32_t numVertices, uint32_t numIndices, uint32_t mVertices_startOffset, 
	uint32_t mIndices_startOffset, std::vector<uint32_t> textures, uint32_t texFlags, Model& model)
	: numVertices(numVertices), numIndices(numIndices), mVertices_startOffset(mVertices_startOffset),
	mIndices_startOffset(mIndices_startOffset), mTextures(textures), texFlags(texFlags), parentModel(model)
{
	//setupMesh();
}


//void Mesh::Draw(Shader shader) {
void Mesh::Draw() {
	unsigned int diffuseNum = 1;
	unsigned int specNum = 1;
	unsigned int norNum = 1;
	unsigned int heightNum = 1;
	unsigned int shininessNum = 1;
	unsigned int emissiveNum = 1;

	//bind textures for this mesh
	for (unsigned int i = 0; i < mTextures.size(); ++i) {
		//glActiveTexture(GL_TEXTURE0 + i);
		std::string number;
		std::string name = parentModel.mTexturesLoaded[mTextures[i]].type;
		if (name == "texture_diffuse") {
			number = std::to_string(diffuseNum++);
		}
		else if (name == "texture_specular") {
			number = std::to_string(specNum++);
		}
		else if (name == "texture_normal") {
			number = std::to_string(norNum++);
		}
		else if (name == "texture_height") {
			number = std::to_string(heightNum++);
		}
		else if (name == "texture_shininess") {
			number = std::to_string(shininessNum++);
		}
		else if (name == "texture_emissive") {
			number = std::to_string(emissiveNum++);
		}
		else {
			std::cout << "Not sure which texture type this is: " << name << std::endl;
		}
		//glUniform1i(glGetUniformLocation(shader.program, (name + number).c_str()), i);
		//glBindTexture(GL_TEXTURE_2D, parentModel.mTexturesLoaded[mTextures[i]].id);
	}
	//glUniform1i(glGetUniformLocation(shader.program, "textureFlags"), texFlags);


	//glBindVertexArray(mVAO);
	//glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);

	////always good to practice to set everything back to defaults once configured
	//glBindVertexArray(0);
	//glActiveTexture(GL_TEXTURE0);
}

//
//void Mesh::setupMesh() {
//	//tell gpu we want some buffers
//	glGenVertexArrays(1, &mVAO);
//	glGenBuffers(1, &mVBO);
//	glGenBuffers(1, &mIBO);
//
//	//"any subsequent VBO, EBO(IBO), attrib calls will be stored in VAO currently bound"
//	glBindVertexArray(mVAO);
//
//
//	//NOTE: Changed to contiguous vertex and index arrays for path tracer
//	////push all the vertex info
//	//glBindBuffer(GL_ARRAY_BUFFER, mVBO);
//	//glBufferData(GL_ARRAY_BUFFER, mVertices.size() * sizeof(Vertex), &mVertices[0], GL_STATIC_DRAW);
//	////push the vertex index info
//	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
//	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndices.size() * sizeof(uint32_t), &mIndices[0], GL_STATIC_DRAW);
//
//	//push all the vertex info
//	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
//	glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vertex), &parentModel.mVertices[mVertices_startOffset], GL_STATIC_DRAW); 
//	//push the vertex index info
//	////NOTE: PathTracer won't need the relative adjustment 
//	std::vector<uint32_t> relativeIndices(numIndices);
//	for (uint32_t i = 0; i < numIndices; ++i) { relativeIndices[i] = parentModel.mIndices[mIndices_startOffset + i] - mVertices_startOffset; }
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(uint32_t), &relativeIndices[0], GL_STATIC_DRAW);
//	//for (uint32_t i = 0; i < numIndices; ++i) { parentModel.mIndices[mIndices_startOffset + i] -= mVertices_startOffset; }
//	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
//	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(uint32_t), &parentModel.mIndices[mIndices_startOffset], GL_STATIC_DRAW);
//
//
//	//tell the gpu how to access our vertex data and which shader locations those attributes will be mapped to
//	uint32_t location = 0;//pos
//	glEnableVertexAttribArray(location);
//	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
//
//	location = 1;//nor
//	glEnableVertexAttribArray(location);
//	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nor));
//
//	location = 2;//uv
//	glEnableVertexAttribArray(location);
//	glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
//
//	location = 3;//tan
//	glEnableVertexAttribArray(location);
//	glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tan));
//
//	////location = 4;//bitan
//	////glEnableVertexAttribArray(location);
//	////glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitan));
//
//	//always good to practice to set everything back to defaults once configured
//	glBindVertexArray(0);
//}
