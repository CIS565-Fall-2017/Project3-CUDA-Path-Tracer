#pragma once
//#include "mesh.h"

//	Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures, uint32_t texFlags)
//		:mVertices(vertices), mIndices(indices), mTextures(textures) , texFlags(texFlags)
//	{
//		setupMesh();
//	}
//
//	void Mesh::Draw(Shader shader) {
//		unsigned int diffuseNum = 1;
//		unsigned int specNum = 1;
//		unsigned int norNum = 1;
//		unsigned int heightNum = 1;
//		unsigned int shininessNum = 1;
//		unsigned int emissiveNum = 1;
//
//		//bind textures for this mesh
//		for (unsigned int i = 0; i < mTextures.size(); ++i) {
//			glActiveTexture(GL_TEXTURE0 + i);
//			std::string number;
//			std::string name = mTextures[i].type;
//			if (name == "texture_diffuse") {
//				number = std::to_string(diffuseNum++);
//			} else if (name == "texture_specular") {
//				number = std::to_string(specNum++);
//			} else if (name == "texture_normal") {
//				number = std::to_string(norNum++);
//			} else if (name == "texture_height") {
//				number = std::to_string(heightNum++);
//			} else if (name == "texture_shininess") {
//				number = std::to_string(shininessNum++);
//			} else if (name == "texture_emissive") {
//				number = std::to_string(emissiveNum++);
//			} else {
//				std::cout << "Not sure which texture type this is: " << name << std::endl;
//			}
//			glUniform1i(glGetUniformLocation(shader.program, (name + number).c_str()), i);
//			glBindTexture(GL_TEXTURE_2D, mTextures[i].id);
//		}
//		glUniform1i(glGetUniformLocation(shader.program, "textureFlags"), texFlags);
//
//
//		glBindVertexArray(mVAO);
//		glDrawElements(GL_TRIANGLES, mIndices.size(), GL_UNSIGNED_INT, 0);
//
//		//always good to practice to set everything back to defaults once configured
//		glBindVertexArray(0);
//		glActiveTexture(GL_TEXTURE0);
//	}
//
//
//	void Mesh::setupMesh() {
//		//tell gpu we want some buffers
//		glGenVertexArrays(1, &mVAO);
//		glGenBuffers(1, &mVBO);
//		glGenBuffers(1, &mIBO);
//
//		//"any subsequent VBO, EBO(IBO), attrib calls will be stored in VAO currently bound"
//		glBindVertexArray(mVAO);
//
//		//push all the vertex info
//		glBindBuffer(GL_ARRAY_BUFFER, mVBO);
//		glBufferData(GL_ARRAY_BUFFER, mVertices.size() * sizeof(Vertex), &mVertices[0], GL_STATIC_DRAW);
//
//		//push the vertex index info
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
//		glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndices.size() * sizeof(unsigned int), &mIndices[0], GL_STATIC_DRAW);
//
//		//tell the gpu how to access our vertex data and which shader locations those attributes will be mapped to
//		unsigned int location = 0;//pos
//		glEnableVertexAttribArray(location);
//		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
//
//		location = 1;//nor
//		glEnableVertexAttribArray(location);
//		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nor));
//
//		location = 2;//uv
//		glEnableVertexAttribArray(location);
//		glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
//
//		location = 3;//tan
//		glEnableVertexAttribArray(location);
//		glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tan));
//
//		location = 4;//bitan
//		glEnableVertexAttribArray(location);
//		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitan));
//
//		//always good to practice to set everything back to defaults once configured
//		glBindVertexArray(0);
//}