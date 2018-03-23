#pragma once
#include "model.h"
#include "mesh.h"

//#include "shader.h"

//glm
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


//SOIL
#include <SOIL2/SOIL2.h>

//////external
//#include <stb/stb_image.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <map>

//assimp
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

std::map<std::string, uint32_t> textureFlags = {
	{"texture_diffuse",			1 <<  0},
	{"texture_specular",		1 <<  1},
	{"texture_normal",			1 <<  2},
	{"texture_height",			1 <<  3},
	{"texture_shininess",		1 <<  4},
	{"texture_emissive",		1 <<  5},
};


uint8_t TextureFromFile(const char* path, const std::string& dir, int& height, int& width, int& channels, uint8_t*& data);

Model::Model(const std::string& path, const bool mirrored, const float unifscale)
	: mirrored(mirrored) , modelmat(scale(mat4(1.f), vec3(unifscale)))
{
	loadModel(path);
	std::cout << "Num Meshes: " << mMeshes.size();
	for (auto& mesh : mMeshes) {
		std::cout << "\nNum verts: " << mesh.mVertices.size() << "\tNum indices: " << mesh.mIndices.size();
	}
	std::cout << "\n";
}


//void Model::Draw(Shader shader) {
//	for (unsigned int i = 0; i < mMeshes.size(); ++i) {
//		mMeshes[i].Draw(shader);
//	}
//}

void Model::loadModel(const std::string& path) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | 
		aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_GenUVCoords);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
		return;
	}

	mDirectory = path.substr(0, path.find_last_of('/'));

	processNode(scene->mRootNode, scene);
}

void Model::processNode(aiNode* node, const aiScene const* scene) {
	//process each mesh located at current node
	for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
		//the node object only contains indices to retrieve te mesh out of the main mMeshes array in scene
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		mMeshes.push_back(processMesh(mesh, scene));
	}

	//process the children of this node
	for (unsigned int i = 0; i < node->mNumChildren; ++i) {
		processNode(node->mChildren[i], scene);
	}
}

Mesh Model::processMesh(const aiMesh* mesh, const aiScene* scene) {
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	std::vector<uint32_t> texIdx;

	//vertices
	for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
		Vertex vertex;
		vertex.pos = vec3(mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z);
		vertex.nor = vec3(mesh->mNormals[i].x,
			mesh->mNormals[i].y,
			mesh->mNormals[i].z);
		if (mesh->mTextureCoords[0]) {//any texture coords?
			//a vertex can contain up to 8 different texture coords. we'll only use the first set (0)
			vertex.uv = vec2(mesh->mTextureCoords[0][i].x,
				mesh->mTextureCoords[0][i].y);
		} else {
			vertex.uv = vec2(0.f, 0.f);
		}

		//we told it to generate tangent space so lets grab them
		if (mesh->mTangents != nullptr && mesh->mBitangents != nullptr) {
			vertex.tan = vec4(mesh->mTangents[i].x,
				mesh->mTangents[i].y,
				mesh->mTangents[i].z, 1.f);
			vertex.bitan = vec3(mesh->mBitangents[i].x,
				mesh->mBitangents[i].y,
				mesh->mBitangents[i].z);
			//eric lengyel 3d math book 3rd ed. pg 185
			vertex.tan.w = (dot(cross(vertex.nor, vec3(vertex.tan)), vertex.bitan) < 0.f ? -1.f : 1.f);
			if (mirrored) { vertex.tan.w *= -1.f; }//why flip it around?
		} else {
			vertex.tan = vec4(0.f, 0.f, 0.f, 1.f);
			vertex.bitan = vec3(0.f, 0.f, 0.f);
		}
		vertices.push_back(vertex);
	}

	//indices
	for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
		aiFace face = mesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; ++j) {
			indices.push_back(face.mIndices[j]);
		}
	}

	/* Materials: we assume a convention for sampler names in the shaders.
	each diffuse texture should be named as "texture_diffuseN" where
	N is a number from 1 to MAX_SAMPLER_NUMBER
	diffuse: texture_diffuseN
	specular: texture_specularN
	normal: texture_normalN
	*/
	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

	//diffuse maps
	const std::vector<uint32_t> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE);
	texIdx.insert(texIdx.end(), diffuseMaps.begin(), diffuseMaps.end());

	//spec maps
	const std::vector<uint32_t> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR);
	texIdx.insert(texIdx.end(), specularMaps.begin(), specularMaps.end());

	//normal maps
	const std::vector<uint32_t> normalMaps = loadMaterialTextures(material, aiTextureType_NORMALS);
	texIdx.insert(texIdx.end(), normalMaps.begin(), normalMaps.end());

	//height maps
	const std::vector<uint32_t> heightMaps = loadMaterialTextures(material, aiTextureType_HEIGHT);
	texIdx.insert(texIdx.end(), heightMaps.begin(), heightMaps.end());

	//shininess(gloss) maps
	const std::vector<uint32_t> shininessMaps = loadMaterialTextures(material, aiTextureType_SHININESS);
	texIdx.insert(texIdx.end(), shininessMaps.begin(), shininessMaps.end());

	//emissive maps
	const std::vector<uint32_t> emissiveMaps = loadMaterialTextures(material, aiTextureType_EMISSIVE);
	texIdx.insert(texIdx.end(), emissiveMaps.begin(), emissiveMaps.end());

	uint32_t texFlags = 0;
	//for (auto& texture : texIdx) {
	for (auto& idx : texIdx) {
		texFlags |= textureFlags.at(mTexturesLoaded[idx].type);
	}

	//std::move this, std::move the args in the constructor
	return Mesh(vertices, indices, texIdx, texFlags, *this);
}

std::vector<uint32_t> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type) {
	std::string typeName;
	if (type == aiTextureType_DIFFUSE)			typeName = "texture_diffuse";
	else if (type == aiTextureType_SPECULAR)	typeName = "texture_specular";
	else if (type == aiTextureType_NORMALS)		typeName = "texture_normal";
	else if (type == aiTextureType_HEIGHT)		typeName = "texture_normal";//treat height as normal(convert outside of program)
	else if (type == aiTextureType_SHININESS)	typeName = "texture_shininess";//aka gloss, spec exponent(greyscale or per channel)
	else if (type == aiTextureType_EMISSIVE)	typeName = "texture_emissive";
	//const int jawn1 = mat->GetTextureCount(aiTextureType_AMBIENT);
	//const int jawn2 = mat->GetTextureCount(aiTextureType_DISPLACEMENT);
	//const int jawn3 = mat->GetTextureCount(aiTextureType_EMISSIVE);
	//const int jawn4 = mat->GetTextureCount(aiTextureType_LIGHTMAP);
	//const int jawn5 = mat->GetTextureCount(aiTextureType_OPACITY);
	//const int jawn6 = mat->GetTextureCount(aiTextureType_REFLECTION);
	//const int jawn7 = mat->GetTextureCount(aiTextureType_UNKNOWN);
	std::vector<uint32_t> textures;
	for (unsigned int i = 0; i < mat->GetTextureCount(type); ++i) {
		aiString str;
		mat->GetTexture(type, i, &str);
		//check if texture was loaded before, if so continue
		bool skip = false;
		for (unsigned int j = 0; j < mTexturesLoaded.size(); ++j) {
			if (std::strcmp(mTexturesLoaded[j].path.C_Str(), str.C_Str()) == 0) {
				textures.push_back(j);
				skip = true;
				break;
			}
		}

		if (skip) { continue; }
		//don't support height. mtl files only have designations for height maps. usually just assign normal maps to its height map field.
		//typeName = typeName == std::string("texture_height") ? std::string("texture_normal") : typeName;

		Texture texture;
		texture.id = TextureFromFile(str.C_Str(), mDirectory, texture.width, texture.height, texture.channels, texture.data);
		texture.type = typeName;
		texture.path = str;
		mTexturesLoaded.push_back(texture);
		//textures.push_back(texture);
		textures.push_back(mTexturesLoaded.size()-1);
	}
	return textures;
}

uint8_t TextureFromFile(const char* path, const std::string& directory, 
	int& width, int& height, int& channels, uint8_t*& data)
{
	std::string filename(path);
	filename = directory + '/' + filename;


	////determine if dds filetype
	//const bool isDDS = filename.substr(filename.find_last_of(".") + 1) == "dds" ? true : false;
	//uint32_t texFlags = SOIL_FLAG_MIPMAPS | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_TEXTURE_REPEATS;
	//texFlags = isDDS ? texFlags | SOIL_FLAG_DDS_LOAD_DIRECT | SOIL_FLAG_INVERT_Y | SOIL_FLAG_COMPRESS_TO_DXT : texFlags;

	//////GLuint textureID = 0;
	uint32_t textureID = 0;
	//textureID = SOIL_load_OGL_texture(filename.c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, texFlags);
	///* check for an error during the load process */
	//if (0 == textureID) { printf("SOIL loading error: '%s'\n", SOIL_last_result()); return -1; } else { return textureID; }


	data = SOIL_load_image(filename.c_str(), &width, &height, &channels, SOIL_LOAD_AUTO);
	if (data) {
		//save texture data to model textures
	} else {
		std::cout << "Texture failed to load: " << path << std::endl;
	}
	return textureID;


	//////////////////////////
	//////NON SOIL METHOD/////
	//////////////////////////
	//unsigned int textureID;
	//glGenTextures(1, &textureID);
	//int width, height, nrComponents;
	//unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);


	//if (data) {
	//	GLenum format;
	//	if (1 == nrComponents) {
	//		format = GL_RED;
	//	} else if (3 == nrComponents) {
	//		format = GL_RGB;
	//	} else if (4 == nrComponents) {
	//		format = GL_RGBA;
	//	}

	//	glBindTexture(GL_TEXTURE_2D, textureID);
	//	glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
	//	glGenerateMipmap(GL_TEXTURE_2D);
	//	
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//} else {
	//	std::cout << "Texture failed to load: " << path << std::endl;
	//}
	//stbi_image_free(data);
	//return textureID;

}

//////////////////////////
////  SOIL EXAMPLES  //////
//////////////////////////
///* load an image file directly as a new OpenGL texture */
//GLuint tex_2d = SOIL_load_OGL_texture
//	(
//		"img.png",
//		SOIL_LOAD_AUTO,
//		SOIL_CREATE_NEW_ID,
//		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
//	);
//	
///* check for an error during the load process */
//if( 0 == tex_2d )
//{
//	printf( "SOIL loading error: '%s'\n", SOIL_last_result() );
//}
//
///* load another image, but into the same texture ID, overwriting the last one */
//tex_2d = SOIL_load_OGL_texture
//	(
//		"some_other_img.dds",
//		SOIL_LOAD_AUTO,
//		tex_2d,
//		SOIL_FLAG_DDS_LOAD_DIRECT
//	);
//	
///* load 6 images into a new OpenGL cube map, forcing RGB */
//GLuint tex_cube = SOIL_load_OGL_cubemap
//	(
//		"xp.jpg",
//		"xn.jpg",
//		"yp.jpg",
//		"yn.jpg",
//		"zp.jpg",
//		"zn.jpg",
//		SOIL_LOAD_RGB,
//		SOIL_CREATE_NEW_ID,
//		SOIL_FLAG_MIPMAPS
//	);
//	
///* load and split a single image into a new OpenGL cube map, default format */
///* face order = East South West North Up Down => "ESWNUD", case sensitive! */
//GLuint single_tex_cube = SOIL_load_OGL_single_cubemap
//	(
//		"split_cubemap.png",
//		"EWUDNS",
//		SOIL_LOAD_AUTO,
//		SOIL_CREATE_NEW_ID,
//		SOIL_FLAG_MIPMAPS
//	);
//	
///* actually, load a DDS cubemap over the last OpenGL cube map, default format */
///* try to load it directly, but give the order of the faces in case that fails */
///* the DDS cubemap face order is pre-defined as SOIL_DDS_CUBEMAP_FACE_ORDER */
//single_tex_cube = SOIL_load_OGL_single_cubemap
//	(
//		"overwrite_cubemap.dds",
//		SOIL_DDS_CUBEMAP_FACE_ORDER,
//		SOIL_LOAD_AUTO,
//		single_tex_cube,
//		SOIL_FLAG_MIPMAPS | SOIL_FLAG_DDS_LOAD_DIRECT
//	);
//	
///* load an image as a heightmap, forcing greyscale (so channels should be 1) */
//int width, height, channels;
//unsigned char *ht_map = SOIL_load_image
//	(
//		"terrain.tga",
//		&width, &height, &channels,
//		SOIL_LOAD_L
//	);
//	
///* save that image as another type */
//int save_result = SOIL_save_image
//	(
//		"new_terrain.dds",
//		SOIL_SAVE_TYPE_DDS,
//		width, height, channels,
//		ht_map
//	);
//	
///* save a screenshot of your awesome OpenGL game engine, running at 1024x768 */
//save_result = SOIL_save_screenshot
//	(
//		"awesomenessity.bmp",
//		SOIL_SAVE_TYPE_BMP,
//		0, 0, 1024, 768
//	);
//
///* loaded a file via PhysicsFS, need to decompress the image from RAM, */
///* where it's in a buffer: unsigned char *image_in_RAM */
//GLuint tex_2d_from_RAM = SOIL_load_OGL_texture_from_memory
//	(
//		image_in_RAM,
//		image_in_RAM_bytes,
//		SOIL_LOAD_AUTO,
//		SOIL_CREATE_NEW_ID,
//		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_COMPRESS_TO_DXT
//	);
//	
///* done with the heightmap, free up the RAM */
//SOIL_free_image_data( ht_map );
