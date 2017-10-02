#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstddef>

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stack>

#define SAH_SUBDIV 15

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;

    char* fname = (char*)filename.c_str();
    fp_in.open(fname);

    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }

    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0)
			{
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) 
			{
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) 
			{
                loadCamera();
                cout << " " << endl;
            }
			else if (strcmp(tokens[0].c_str(), "FILM") == 0)
			{
				loadFilm();
				cout << " " << endl;
			}
        }
    }

	// After loading, initialize
	this->initialize();
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
			else if (strcmp(line.c_str(), "mesh") == 0) {
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
			}
        }

		if (newGeom.type == MESH)
		{
			utilityCore::safeGetline(fp_in, line);
			if (!line.empty() && fp_in.good()) 
			{
				vector<string> tokens = utilityCore::tokenizeString(line);
				if (strcmp(tokens[0].c_str(), "FILE") == 0)
				{
					string path = tokens[1];
					newGeom.meshData = loadMesh(path);
					cout << "Loading mesh " << path << endl;
				}
			}
		}

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}
int Scene::loadFilm() {
	cout << "Loading Film ..." << endl;
	RenderState &state = this->state;
	Film &film = state.film;

	string line;
	utilityCore::safeGetline(fp_in, line);

	while (!line.empty() && fp_in.good()) 
	{
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "FILTER_RADIUS") == 0) 
		{
			film.filterRadius = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FILTER_ALPHA") == 0) 
		{
			film.filterAlpha = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "GAMMA") == 0) 
		{
			film.gamma = atof(tokens[1].c_str());
			film.invGamma = 1.f / film.gamma;
		}
		else if (strcmp(tokens[0].c_str(), "EXPOSURE") == 0)
		{
			film.exposure = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "VIGNETTE_START") == 0)
		{
			film.vignetteStart = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "VIGNETTE_END") == 0)
		{
			film.vignetteEnd = atof(tokens[1].c_str());
		}

		utilityCore::safeGetline(fp_in, line);
	}

	cout << "Loaded film!" << endl;
	return 1;
}


TextureDescriptor Scene::loadTexture(string path, bool normalize)
{
	TextureDescriptor desc;
	if (path == "MANDELBROT")
	{
		desc.type = 1;
		desc.valid = 1;
	}
	else
	{
		Texture * tex = nullptr;
		
		if (textureMap.find(path) == textureMap.end())
		{
			tex = new Texture(path, 1.f, normalize);
			this->textures.push_back(tex);
		}
		else
		{
			tex = textureMap[path];
		}

		desc.type = 0;
		desc.index = textures.size() - 1;
		desc.width = tex->GetWidth();
		desc.height = tex->GetHeight();
		desc.valid = 1;
	}

	return desc;
}

void Scene::initialize()
{
	Film & film = state.film;

	glm::vec3 g = glm::vec3(film.gamma);

	for (Material & m : this->materials)
	{
		m.color = glm::pow(m.color, g);
		m.specular.color = glm::pow(m.specular.color, g);
	}

	for (int i = 0; i < geoms.size(); i++)
	{
		if (materials[geoms[i].materialid].emittance > 0.f)
			lights.push_back(i);
	}

	cout << "Lights: " << lights.size() << endl;
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
	Film &film = state.film;
    float fovy;

	camera.aperture = 0.f;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        } 
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
		else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } 
		else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
		else if (strcmp(tokens[0].c_str(), "APERTURE") == 0) {
			state.camera.aperture = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DISTANCE") == 0) {
			state.camera.focalDistance = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "BOKEH") == 0) {
			state.camera.bokehTexture = loadTexture(tokens[1], true);
		}

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec4());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

		string line;
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) 
		{

            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } 
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) 
			{
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } 
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0)
			{
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } 
			else if (strcmp(tokens[0].c_str(), "REFL") == 0)
			{
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } 
			else if (strcmp(tokens[0].c_str(), "REFR") == 0)
			{
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } 
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0)
			{
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } 
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) 
			{
                newMaterial.emittance = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "TEX_DIFFUSE") == 0) 
			{
				newMaterial.diffuseTexture = loadTexture(tokens[1], false);
			}
			else if (strcmp(tokens[0].c_str(), "TEX_SPECULAR") == 0) 
			{
				newMaterial.specularTexture = loadTexture(tokens[1], false);
			}
			else if (strcmp(tokens[0].c_str(), "TEX_NORMAL") == 0)
			{
				newMaterial.normalTexture = loadTexture(tokens[1], false);
			}
			else if (strcmp(tokens[0].c_str(), "TRANSLUCENCE") == 0)
			{
				newMaterial.translucence = atof(tokens[1].c_str());
			}


			utilityCore::safeGetline(fp_in, line);
		}

        materials.push_back(newMaterial);
        return 1;
    }
}

MeshDescriptor Scene::loadMesh(string & filename)
{
	if (meshMap.find(filename) != meshMap.end())
	{
		Mesh * mesh = meshMap[filename];

		MeshDescriptor desc;
		desc.offset = -1;

		for (int i = 0; i < meshes.size(); i++)
			if (meshes[i] == mesh)
				desc.offset = i;

		cout << "Reusing mesh " + filename << endl;
		return desc;
	}

	std::vector<tinyobj::shape_t> shapes; std::vector<tinyobj::material_t> materials;
	std::string errors = tinyobj::LoadObj(shapes, materials, filename.c_str());
	std::cout << errors << std::endl;

	std::vector<Triangle*> triangles;

	if (errors.size() == 0)
	{
		//Read the information from the vector of shape_ts
		for (unsigned int i = 0; i < shapes.size(); i++)
		{
			std::vector<float> &positions = shapes[i].mesh.positions;
			std::vector<float> &normals = shapes[i].mesh.normals;
			std::vector<float> &uvs = shapes[i].mesh.texcoords;
			std::vector<unsigned int> &indices = shapes[i].mesh.indices;
			for (unsigned int j = 0; j < indices.size(); j += 3)
			{
				glm::vec3 p1(positions[indices[j] * 3], positions[indices[j] * 3 + 1], positions[indices[j] * 3 + 2]);
				glm::vec3 p2(positions[indices[j + 1] * 3], positions[indices[j + 1] * 3 + 1], positions[indices[j + 1] * 3 + 2]);
				glm::vec3 p3(positions[indices[j + 2] * 3], positions[indices[j + 2] * 3 + 1], positions[indices[j + 2] * 3 + 2]);

				glm::vec3 min = glm::min(p1, glm::min(p2, p3));
				glm::vec3 max = glm::max(p1, glm::max(p2, p3));
				AABB bounds = AABB(min - glm::vec3(glm::epsilon<float>()), max + glm::vec3(glm::epsilon<float>()));
								
				Triangle * triangle = new Triangle();
				triangle->p1 = p1;
				triangle->p2 = p2;
				triangle->p3 = p3;
				triangle->bounds = bounds;

				if (normals.size() > 0)
				{
					glm::vec3 n1(normals[indices[j] * 3], normals[indices[j] * 3 + 1], normals[indices[j] * 3 + 2]);
					glm::vec3 n2(normals[indices[j + 1] * 3], normals[indices[j + 1] * 3 + 1], normals[indices[j + 1] * 3 + 2]);
					glm::vec3 n3(normals[indices[j + 2] * 3], normals[indices[j + 2] * 3 + 1], normals[indices[j + 2] * 3 + 2]);
					
					triangle->n1 = n1;
					triangle->n2 = n2;
					triangle->n3 = n3;
				}
				if (uvs.size() > 0)
				{
					glm::vec2 t1(uvs[indices[j] * 2], uvs[indices[j] * 2 + 1]);
					glm::vec2 t2(uvs[indices[j + 1] * 2], uvs[indices[j + 1] * 2 + 1]);
					glm::vec2 t3(uvs[indices[j + 2] * 2], uvs[indices[j + 2] * 2 + 1]);
				/*	t->uvs[0] = t1;
					t->uvs[1] = t2;
					t->uvs[2] = t3;*/
				}

				triangles.push_back(triangle);
			}
		}

		std::cout << "" << std::endl;
	}
	else
	{
		//An error loading the OBJ occurred!
		std::cout << errors << std::endl;
	}

	Mesh * mesh = new Mesh(15, 2, triangles);
	this->meshes.push_back(mesh);
	meshMap[filename] = mesh;

	// Needs to be transformed later
	MeshDescriptor desc;
	desc.offset = this->meshes.size() - 1;

	cout << "Loaded mesh with " << triangles.size() << " triangles" << endl;
	return desc;
}

glm::vec3 axisPlaneNormals[] = { glm::vec3(1.f,0,0), glm::vec3(0,1.f,0), glm::vec3(0,0,1.f) };

Mesh::MeshNode::MeshNode(const std::vector<Triangle *>& originalTriangles, glm::vec3 min, glm::vec3 max, int depth, int maxDepth, int threshold)
{
	glm::vec3 extent = glm::abs(max - min);

	if (extent.x > extent.y && extent.x > extent.z)
		this->axis = 0;
	else if (extent.y > extent.x && extent.y > extent.z)
		this->axis = 1;
	else
		this->axis = 2;

	this->left = nullptr;
	this->right = nullptr;
	this->split = 0;
	this->BuildNode(originalTriangles, min, max, depth, maxDepth, threshold);
	this->parentOffset = -1;
}

Mesh::MeshNode::~MeshNode()
{
	if (this->left != nullptr)
		delete left;

	if (this->right != nullptr)
		delete right;
}

void Mesh::MeshNode::BuildNode(const std::vector<Triangle *>& triangles, const glm::vec3 &minVector, const glm::vec3 &maxVector, int depth, int maxDepth, int threshold)
{
	if (triangles.size() > threshold && depth < maxDepth)
	{
		glm::vec3 axisNormal = axisPlaneNormals[axis];

		float minAxis = minVector[axis];
		float maxAxis = maxVector[axis];

		// If axis cannot be subdivided, we stop, to prevent jumping
		// between axis indefinitely
		if (glm::abs(minAxis - maxAxis) < glm::epsilon<float>())
		{
			this->nodeTriangles = triangles;
			this->left = nullptr;
			this->right = nullptr;
			return;
		}

		split = GetSplitPoint(triangles, minVector[axis], maxVector[axis]);

		std::vector<Triangle*> leftShapes;
		std::vector<Triangle*> rightShapes;

		for (int i = 0; i < triangles.size(); i++)
		{
			Triangle * tri = triangles[i];
			AABB bounds = tri->bounds;

			float p = bounds.center[axis];

			// If shape position is on right, surely its on right node
			if (p > split) {
				rightShapes.push_back(tri);

				// But if bounding box collides with plane, add on left
				// node
				float min = bounds.min[axis];

				if (min <= split)
					leftShapes.push_back(tri);

			}
			else {
				leftShapes.push_back(tri);

				// But if bounding box collides with plane, add on right
				// node
				float max = bounds.max[axis];

				if (max >= split)
					rightShapes.push_back(tri);
			}
		}

		glm::vec3 leftMax = maxVector - (axisNormal * glm::abs(maxVector[axis] - split));
		glm::vec3 rightMin = minVector + (axisNormal * glm::abs(split - minVector[axis]));

		this->left = new MeshNode(leftShapes, minVector, leftMax, depth + 1, maxDepth, threshold);
		this->right = new MeshNode(rightShapes, rightMin, maxVector, depth + 1, maxDepth, threshold);
	}
	else
	{
		this->nodeTriangles = triangles;
		this->left = nullptr;
		this->right = nullptr;
	}
}

float Mesh::MeshNode::CostFunction(float split, const std::vector<Triangle *>& triangles, float minAxis, float maxAxis)
{
	int leftCount = 0;
	int rightCount = 0;

	for (int i = 0; i < triangles.size(); i++)
	{
		Triangle * tri = triangles[i];
		AABB bounds = tri->bounds;

		float p = bounds.center[axis];

		// If shape position is on right, surely its on right node
		if (p > split) {
			rightCount++;

			// But if bounding box collides with plane, add on left
			// node
			float min = bounds.min[axis];

			if (min <= split)
				leftCount++;

		}
		else {
			leftCount++;

			// But if bounding box collides with plane, add on right
			// node
			float max = bounds.max[axis];

			if (max >= split)
				rightCount++;
		}
	}

	// Here we simplify Surface area by using just the size on the split
	// axis
	float leftSize = split - minAxis;
	float rightSize = maxAxis - split;

	return (leftSize * leftCount) + (rightSize * rightCount);
}

float Mesh::MeshNode::GetSplitPoint(const std::vector<Triangle *> &triangles, float minAxis, float maxAxis)
{
	// Spatial median
	float center = (maxAxis + minAxis) * .5f;

	// Object median
	float objMedian = 0;

	for (int i = 0; i < triangles.size(); i++)
		objMedian += triangles[i]->bounds.center[axis];

	objMedian /= triangles.size();

	float step = (center - objMedian) / SAH_SUBDIV;

	float minCost = std::numeric_limits<float>::infinity();
	float result = objMedian;

	if (glm::abs(step) > glm::epsilon<float>())
	{
		// i is the proposed split point
		for (float i = objMedian; i < center; i += step)
		{
			float cost = CostFunction(i, triangles, minAxis, maxAxis);

			if (minCost > cost)
			{
				minCost = cost;
				result = i;
			}
		}
	}

	return result;
}

int Mesh::MeshNode::GetNodeCount()
{
	if (IsLeaf())
		return 1;
	else
		return 1 + left->GetNodeCount() + right->GetNodeCount();
}

int Mesh::MeshNode::TriangleCount()
{
	if (IsLeaf())
		return this->nodeTriangles.size();
	else
		return left->TriangleCount() + right->TriangleCount();
}

int Mesh::MeshNode::GetDepth()
{
	if (IsLeaf())
		return 1;

	return glm::max(left->GetDepth(), right->GetDepth()) + 1;
}

bool Mesh::MeshNode::IsLeaf()
{
	return left == nullptr && right == nullptr;
}

Mesh::Mesh(int maxDepth, int maxLeafSize, std::vector<Triangle*>& triangles) : maxDepth(maxDepth), maxLeafSize(maxLeafSize), root(nullptr), compactNodes(nullptr), triangles(triangles)
{
}

Mesh::~Mesh()
{
	if (this->root != nullptr)
		delete this->root;

	if (this->compactNodes != nullptr)
		delete[] this->compactNodes;
}

AABB Mesh::CalculateAABB()
{
	AABB bounds;

	if (triangles.size() > 0)
		bounds = triangles[0]->bounds;

	for (int i = 1; i < triangles.size(); i++)
		bounds = bounds.Encapsulate(triangles[i]->bounds);

	return bounds;
}

void Mesh::Build()
{
	for (int i = 0; i < triangles.size(); i++)
	{
		Triangle * t = triangles[i];
		glm::vec3 min = glm::min(t->p1, glm::min(t->p2, t->p3));
		glm::vec3 max = glm::max(t->p1, glm::max(t->p2, t->p3));
		t->bounds = AABB(min - glm::vec3(glm::epsilon<float>()), max + glm::vec3(glm::epsilon<float>()));
	}

	meshBounds = this->CalculateAABB();
	this->root = new MeshNode(triangles, meshBounds.min, meshBounds.max, 0, this->maxDepth, this->maxLeafSize);

	std::cout << this->root->GetDepth() << " tree depth" << std::endl;
	std::cout << this->root->GetNodeCount() << " tree nodes" << std::endl;
	std::cout << this->root->TriangleCount() << " total elements in tree" << std::endl;

	this->Compact();
}

void Mesh::Compact()
{
	int nodeCount = this->root->GetNodeCount();
	int elements = this->root->TriangleCount();

	// Left, right, split, axis, primitiveCount
	int nodeSize = 4 + 4 + 4 + 4 + 4;
	int elementSize = sizeof(CompactTriangle);

	this->compactDataSize = (nodeCount * nodeSize) + (elements * elementSize);
	this->compactNodes = (int*)std::malloc(compactDataSize);

	int mb = compactDataSize / (1024 * 1024);
	std::cout << "Compact mesh kd-tree memory size: " << mb << " megabytes" << std::endl;

	if (this->compactNodes == nullptr)
		std::cerr << "Could not allocate " << mb << " megabytes" << std::endl;

	std::stack<MeshNode*> stack;
	stack.push(this->root);

	int offset = 0;

	// For compaction, we need to remember the original placement of nodes in parent
	// nodes because we can't really know the offset until we build everything.
	while (!stack.empty())
	{
		MeshNode * node = stack.top();
		stack.pop();

		if (node == nullptr)
			continue;

		// If this node is the child of a parent, let's set the current offset
		if (node->parentOffset != -1)
			compactNodes[node->parentOffset] = offset;

		int baseOffset = offset;

		CompactNode * cNode = (CompactNode*)(compactNodes + offset);
		cNode->leftNode = -1;
		cNode->rightNode = -1;
		cNode->split = node->split;
		cNode->axis = node->axis;

		if (node->IsLeaf())
		{
			cNode->primitiveCount = node->nodeTriangles.size();
			offset += 5;

			float * triData = (float*)compactNodes;
			int triCount = node->nodeTriangles.size();

			for (int i = 0; i < triCount; i++)
			{
				Triangle * triangle = node->nodeTriangles[i];

				glm::vec3 e1 = triangle->p2 - triangle->p1;
				glm::vec3 e2 = triangle->p3 - triangle->p1;
				glm::vec3 p = triangle->p1;

				// e1
				triData[offset++] = e1.x;
				triData[offset++] = e1.y;
				triData[offset++] = e1.z;

				// e2
				triData[offset++] = e2.x;
				triData[offset++] = e2.y;
				triData[offset++] = e2.z;

				// p1
				triData[offset++] = triangle->p1.x;
				triData[offset++] = triangle->p1.y;
				triData[offset++] = triangle->p1.z;

				// Normals
				triData[offset++] = triangle->n1.x;
				triData[offset++] = triangle->n1.y;
				triData[offset++] = triangle->n1.z;

				triData[offset++] = triangle->n2.x;
				triData[offset++] = triangle->n2.y;
				triData[offset++] = triangle->n2.z;

				triData[offset++] = triangle->n3.x;
				triData[offset++] = triangle->n3.y;
				triData[offset++] = triangle->n3.z;
			}
		}
		else
		{
			cNode->primitiveCount = 0;
			offset += 5;

			node->left->parentOffset = baseOffset;
			node->right->parentOffset = baseOffset + 1;

			stack.push(node->left);
			stack.push(node->right);
		}
	}

	// Now that everything is copied and compacted, we can delete our root
	delete this->root;
	this->root = nullptr;
}

glm::vec3 AABB::aabb[] = { glm::vec3(1, 1, 1),glm::vec3(1, -1, -1), glm::vec3(1, 1, -1), glm::vec3(1, -1, 1),
glm::vec3(-1, 1, 1), glm::vec3(-1, -1, -1), glm::vec3(-1, 1, -1), glm::vec3(-1, -1, 1) };

AABB AABB::Encapsulate(AABB bounds)
{
	glm::vec3 min = glm::min(this->min, bounds.min);
	glm::vec3 max = glm::max(this->max, bounds.max);
	return AABB(min, max);
}

AABB AABB::Transform(const glm::mat4x4 &transform)
{
	// If infinite box, prevent overflowing
	if (min.x == -std::numeric_limits<float>::infinity() || min.y == -std::numeric_limits<float>::infinity() || min.z == -std::numeric_limits<float>::infinity()
		|| max.x == std::numeric_limits<float>::infinity() || max.y == std::numeric_limits<float>::infinity() || max.z == std::numeric_limits<float>::infinity())
	{
		return *this;
	}

	float maxX = -std::numeric_limits<float>::infinity();
	float maxY = -std::numeric_limits<float>::infinity();
	float maxZ = -std::numeric_limits<float>::infinity();

	float minX = std::numeric_limits<float>::infinity();
	float minY = std::numeric_limits<float>::infinity();
	float minZ = std::numeric_limits<float>::infinity();

	glm::vec3 halfSize = (max - center);

	for (int i = 0; i < 8; i++)
	{
		glm::vec3 v = center + (AABB::aabb[i] * halfSize);
		glm::vec3 tPoint = glm::vec3(transform * glm::vec4(v.x, v.y, v.z, 1.));

		maxX = glm::max(tPoint.x, maxX);
		maxY = glm::max(tPoint.y, maxY);
		maxZ = glm::max(tPoint.z, maxZ);

		minX = glm::min(tPoint.x, minX);
		minY = glm::min(tPoint.y, minY);
		minZ = glm::min(tPoint.z, minZ);
	}

	glm::vec3 newMin = glm::vec3(minX, minY, minZ);
	glm::vec3 newMax = glm::vec3(maxX, maxY, maxZ);

	return AABB(newMin, newMax);
}

//
// Copyright 2012-2015, Syoyo Fujita.
//
// Licensed under 2-clause BSD liecense.
//

//
// version 0.9.9: Replace atof() with custom parser.
// version 0.9.8: Fix multi-materials(per-face material ID).
// version 0.9.7: Support multi-materials(per-face material ID) per
// object/group.
// version 0.9.6: Support Ni(index of refraction) mtl parameter.
//                Parse transmittance material parameter correctly.
// version 0.9.5: Parse multiple group name.
//                Add support of specifying the base path to load material file.
// version 0.9.4: Initial suupport of group tag(g)
// version 0.9.3: Fix parsing triple 'x/y/z'
// version 0.9.2: Add more .mtl load support
// version 0.9.1: Add initial .mtl load support
// version 0.9.0: Initial
//

namespace tinyobj {

	struct vertex_index {
		int v_idx, vt_idx, vn_idx;
		vertex_index() {};
		vertex_index(int idx) : v_idx(idx), vt_idx(idx), vn_idx(idx) {};
		vertex_index(int vidx, int vtidx, int vnidx)
			: v_idx(vidx), vt_idx(vtidx), vn_idx(vnidx) {};
	};
	// for std::map
	static inline bool operator<(const vertex_index &a, const vertex_index &b) {
		if (a.v_idx != b.v_idx)
			return (a.v_idx < b.v_idx);
		if (a.vn_idx != b.vn_idx)
			return (a.vn_idx < b.vn_idx);
		if (a.vt_idx != b.vt_idx)
			return (a.vt_idx < b.vt_idx);

		return false;
	}

	struct obj_shape {
		std::vector<float> v;
		std::vector<float> vn;
		std::vector<float> vt;
	};

	static inline bool isSpace(const char c) { return (c == ' ') || (c == '\t'); }

	static inline bool isNewLine(const char c) {
		return (c == '\r') || (c == '\n') || (c == '\0');
	}

	// Make index zero-base, and also support relative index.
	static inline int fixIndex(int idx, int n) {
		if (idx > 0) return idx - 1;
		if (idx == 0) return 0;
		return n + idx; // negative value = relative
	}

	static inline std::string parseString(const char *&token) {
		std::string s;
		token += strspn(token, " \t");
		size_t e = strcspn(token, " \t\r");
		s = std::string(token, &token[e]);
		token += e;
		return s;
	}

	static inline int parseInt(const char *&token) {
		token += strspn(token, " \t");
		int i = atoi(token);
		token += strcspn(token, " \t\r");
		return i;
	}


	// Tries to parse a floating point number located at s.
	//
	// s_end should be a location in the string where reading should absolutely
	// stop. For example at the end of the string, to prevent buffer overflows.
	//
	// Parses the following EBNF grammar:
	//   sign    = "+" | "-" ;
	//   END     = ? anything not in digit ?
	//   digit   = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
	//   integer = [sign] , digit , {digit} ;
	//   decimal = integer , ["." , integer] ;
	//   float   = ( decimal , END ) | ( decimal , ("E" | "e") , integer , END ) ;
	//
	//  Valid strings are for example:
	//   -0	 +3.1417e+2  -0.0E-3  1.0324  -1.41   11e2
	//
	// If the parsing is a success, result is set to the parsed value and true 
	// is returned.
	//
	// The function is greedy and will parse until any of the following happens:
	//  - a non-conforming character is encountered.
	//  - s_end is reached.
	//
	// The following situations triggers a failure:
	//  - s >= s_end.
	//  - parse failure.
	// 
	static bool tryParseDouble(const char *s, const char *s_end, double *result)
	{
		if (s >= s_end)
		{
			return false;
		}

		double mantissa = 0.0;
		// This exponent is base 2 rather than 10.
		// However the exponent we parse is supposed to be one of ten,
		// thus we must take care to convert the exponent/and or the 
		// mantissa to a * 2^E, where a is the mantissa and E is the
		// exponent.
		// To get the final double we will use ldexp, it requires the
		// exponent to be in base 2.
		int exponent = 0;

		// NOTE: THESE MUST BE DECLARED HERE SINCE WE ARE NOT ALLOWED
		// TO JUMP OVER DEFINITIONS.
		char sign = '+';
		char exp_sign = '+';
		char const *curr = s;

		// How many characters were read in a loop. 
		int read = 0;
		// Tells whether a loop terminated due to reaching s_end.
		bool end_not_reached = false;

		/*
		BEGIN PARSING.
		*/

		// Find out what sign we've got.
		if (*curr == '+' || *curr == '-')
		{
			sign = *curr;
			curr++;
		}
		else if (isdigit(*curr)) { /* Pass through. */ }
		else
		{
			goto fail;
		}

		// Read the integer part.
		while ((end_not_reached = (curr != s_end)) && isdigit(*curr))
		{
			mantissa *= 10;
			mantissa += static_cast<int>(*curr - 0x30);
			curr++;	read++;
		}

		// We must make sure we actually got something.
		if (read == 0)
			goto fail;
		// We allow numbers of form "#", "###" etc.
		if (!end_not_reached)
			goto assemble;

		// Read the decimal part.
		if (*curr == '.')
		{
			curr++;
			read = 1;
			while ((end_not_reached = (curr != s_end)) && isdigit(*curr))
			{
				// NOTE: Don't use powf here, it will absolutely murder precision.
				mantissa += static_cast<int>(*curr - 0x30) * pow(10, -read);
				read++; curr++;
			}
		}
		else if (*curr == 'e' || *curr == 'E') {}
		else
		{
			goto assemble;
		}

		if (!end_not_reached)
			goto assemble;

		// Read the exponent part.
		if (*curr == 'e' || *curr == 'E')
		{
			curr++;
			// Figure out if a sign is present and if it is.
			if ((end_not_reached = (curr != s_end)) && (*curr == '+' || *curr == '-'))
			{
				exp_sign = *curr;
				curr++;
			}
			else if (isdigit(*curr)) { /* Pass through. */ }
			else
			{
				// Empty E is not allowed.
				goto fail;
			}

			read = 0;
			while ((end_not_reached = (curr != s_end)) && isdigit(*curr))
			{
				exponent *= 10;
				exponent += static_cast<int>(*curr - 0x30);
				curr++;	read++;
			}
			exponent *= (exp_sign == '+' ? 1 : -1);
			if (read == 0)
				goto fail;
		}

	assemble:
		*result = (sign == '+' ? 1 : -1) * ldexp(mantissa * pow(5, exponent), exponent);
		return true;
	fail:
		return false;
	}
	static inline float parseFloat(const char *&token) {
		token += strspn(token, " \t");
#ifdef TINY_OBJ_LOADER_OLD_FLOAT_PARSER
		float f = (float)atof(token);
		token += strcspn(token, " \t\r");
#else
		const char *end = token + strcspn(token, " \t\r");
		double val = 0.0;
		tryParseDouble(token, end, &val);
		float f = static_cast<float>(val);
		token = end;
#endif
		return f;
	}


	static inline void parseFloat2(float &x, float &y, const char *&token) {
		x = parseFloat(token);
		y = parseFloat(token);
	}

	static inline void parseFloat3(float &x, float &y, float &z,
		const char *&token) {
		x = parseFloat(token);
		y = parseFloat(token);
		z = parseFloat(token);
	}

	// Parse triples: i, i/j/k, i//k, i/j
	static vertex_index parseTriple(const char *&token, int vsize, int vnsize,
		int vtsize) {
		vertex_index vi(-1);

		vi.v_idx = fixIndex(atoi(token), vsize);
		token += strcspn(token, "/ \t\r");
		if (token[0] != '/') {
			return vi;
		}
		token++;

		// i//k
		if (token[0] == '/') {
			token++;
			vi.vn_idx = fixIndex(atoi(token), vnsize);
			token += strcspn(token, "/ \t\r");
			return vi;
		}

		// i/j/k or i/j
		vi.vt_idx = fixIndex(atoi(token), vtsize);
		token += strcspn(token, "/ \t\r");
		if (token[0] != '/') {
			return vi;
		}

		// i/j/k
		token++; // skip '/'
		vi.vn_idx = fixIndex(atoi(token), vnsize);
		token += strcspn(token, "/ \t\r");
		return vi;
	}

	static unsigned int
		updateVertex(std::map<vertex_index, unsigned int> &vertexCache,
			std::vector<float> &positions, std::vector<float> &normals,
			std::vector<float> &texcoords,
			const std::vector<float> &in_positions,
			const std::vector<float> &in_normals,
			const std::vector<float> &in_texcoords, const vertex_index &i) {
		const std::map<vertex_index, unsigned int>::iterator it = vertexCache.find(i);

		if (it != vertexCache.end()) {
			// found cache
			return it->second;
		}

		assert(in_positions.size() > (unsigned int)(3 * i.v_idx + 2));

		positions.push_back(in_positions[3 * i.v_idx + 0]);
		positions.push_back(in_positions[3 * i.v_idx + 1]);
		positions.push_back(in_positions[3 * i.v_idx + 2]);

		if (i.vn_idx >= 0) {
			normals.push_back(in_normals[3 * i.vn_idx + 0]);
			normals.push_back(in_normals[3 * i.vn_idx + 1]);
			normals.push_back(in_normals[3 * i.vn_idx + 2]);
		}

		if (i.vt_idx >= 0) {
			texcoords.push_back(in_texcoords[2 * i.vt_idx + 0]);
			texcoords.push_back(in_texcoords[2 * i.vt_idx + 1]);
		}

		unsigned int idx = static_cast<unsigned int>(positions.size() / 3 - 1);
		vertexCache[i] = idx;

		return idx;
	}

	void InitMaterial(material_t &material) {
		material.name = "";
		material.ambient_texname = "";
		material.diffuse_texname = "";
		material.specular_texname = "";
		material.normal_texname = "";
		for (int i = 0; i < 3; i++) {
			material.ambient[i] = 0.f;
			material.diffuse[i] = 0.f;
			material.specular[i] = 0.f;
			material.transmittance[i] = 0.f;
			material.emission[i] = 0.f;
		}
		material.illum = 0;
		material.dissolve = 1.f;
		material.shininess = 1.f;
		material.ior = 1.f;
		material.unknown_parameter.clear();
	}

	static bool exportFaceGroupToShape(
		shape_t &shape, std::map<vertex_index, unsigned int> vertexCache,
		const std::vector<float> &in_positions,
		const std::vector<float> &in_normals,
		const std::vector<float> &in_texcoords,
		const std::vector<std::vector<vertex_index> > &faceGroup,
		const int material_id, const std::string &name, bool clearCache) {
		if (faceGroup.empty()) {
			return false;
		}

		// Flatten vertices and indices
		for (size_t i = 0; i < faceGroup.size(); i++) {
			const std::vector<vertex_index> &face = faceGroup[i];

			vertex_index i0 = face[0];
			vertex_index i1(-1);
			vertex_index i2 = face[1];

			size_t npolys = face.size();

			// Polygon -> triangle fan conversion
			for (size_t k = 2; k < npolys; k++) {
				i1 = i2;
				i2 = face[k];

				unsigned int v0 = updateVertex(
					vertexCache, shape.mesh.positions, shape.mesh.normals,
					shape.mesh.texcoords, in_positions, in_normals, in_texcoords, i0);
				unsigned int v1 = updateVertex(
					vertexCache, shape.mesh.positions, shape.mesh.normals,
					shape.mesh.texcoords, in_positions, in_normals, in_texcoords, i1);
				unsigned int v2 = updateVertex(
					vertexCache, shape.mesh.positions, shape.mesh.normals,
					shape.mesh.texcoords, in_positions, in_normals, in_texcoords, i2);

				shape.mesh.indices.push_back(v0);
				shape.mesh.indices.push_back(v1);
				shape.mesh.indices.push_back(v2);

				shape.mesh.material_ids.push_back(material_id);
			}
		}

		shape.name = name;

		if (clearCache)
			vertexCache.clear();

		return true;
	}

	std::string LoadMtl(std::map<std::string, int> &material_map,
		std::vector<material_t> &materials,
		std::istream &inStream) {
		std::stringstream err;

		material_t material;

		int maxchars = 8192;             // Alloc enough size.
		std::vector<char> buf(maxchars); // Alloc enough size.
		while (inStream.peek() != -1) {
			inStream.getline(&buf[0], maxchars);

			std::string linebuf(&buf[0]);

			// Trim newline '\r\n' or '\n'
			if (linebuf.size() > 0) {
				if (linebuf[linebuf.size() - 1] == '\n')
					linebuf.erase(linebuf.size() - 1);
			}
			if (linebuf.size() > 0) {
				if (linebuf[linebuf.size() - 1] == '\r')
					linebuf.erase(linebuf.size() - 1);
			}

			// Skip if empty line.
			if (linebuf.empty()) {
				continue;
			}

			// Skip leading space.
			const char *token = linebuf.c_str();
			token += strspn(token, " \t");

			assert(token);
			if (token[0] == '\0')
				continue; // empty line

			if (token[0] == '#')
				continue; // comment line

						  // new mtl
			if ((0 == strncmp(token, "newmtl", 6)) && isSpace((token[6]))) {
				// flush previous material.
				if (!material.name.empty()) {
					material_map.insert(
						std::pair<std::string, int>(material.name, static_cast<int>(materials.size())));
					materials.push_back(material);
				}

				// initial temporary material
				InitMaterial(material);

				// set new mtl name
				char namebuf[4096];
				token += 7;
#ifdef _MSC_VER
				sscanf_s(token, "%s", namebuf);
#else
				sscanf(token, "%s", namebuf);
#endif
				material.name = namebuf;
				continue;
			}

			// ambient
			if (token[0] == 'K' && token[1] == 'a' && isSpace((token[2]))) {
				token += 2;
				float r, g, b;
				parseFloat3(r, g, b, token);
				material.ambient[0] = r;
				material.ambient[1] = g;
				material.ambient[2] = b;
				continue;
			}

			// diffuse
			if (token[0] == 'K' && token[1] == 'd' && isSpace((token[2]))) {
				token += 2;
				float r, g, b;
				parseFloat3(r, g, b, token);
				material.diffuse[0] = r;
				material.diffuse[1] = g;
				material.diffuse[2] = b;
				continue;
			}

			// specular
			if (token[0] == 'K' && token[1] == 's' && isSpace((token[2]))) {
				token += 2;
				float r, g, b;
				parseFloat3(r, g, b, token);
				material.specular[0] = r;
				material.specular[1] = g;
				material.specular[2] = b;
				continue;
			}

			// transmittance
			if (token[0] == 'K' && token[1] == 't' && isSpace((token[2]))) {
				token += 2;
				float r, g, b;
				parseFloat3(r, g, b, token);
				material.transmittance[0] = r;
				material.transmittance[1] = g;
				material.transmittance[2] = b;
				continue;
			}

			// ior(index of refraction)
			if (token[0] == 'N' && token[1] == 'i' && isSpace((token[2]))) {
				token += 2;
				material.ior = parseFloat(token);
				continue;
			}

			// emission
			if (token[0] == 'K' && token[1] == 'e' && isSpace(token[2])) {
				token += 2;
				float r, g, b;
				parseFloat3(r, g, b, token);
				material.emission[0] = r;
				material.emission[1] = g;
				material.emission[2] = b;
				continue;
			}

			// shininess
			if (token[0] == 'N' && token[1] == 's' && isSpace(token[2])) {
				token += 2;
				material.shininess = parseFloat(token);
				continue;
			}

			// illum model
			if (0 == strncmp(token, "illum", 5) && isSpace(token[5])) {
				token += 6;
				material.illum = parseInt(token);
				continue;
			}

			// dissolve
			if ((token[0] == 'd' && isSpace(token[1]))) {
				token += 1;
				material.dissolve = parseFloat(token);
				continue;
			}
			if (token[0] == 'T' && token[1] == 'r' && isSpace(token[2])) {
				token += 2;
				material.dissolve = parseFloat(token);
				continue;
			}

			// ambient texture
			if ((0 == strncmp(token, "map_Ka", 6)) && isSpace(token[6])) {
				token += 7;
				material.ambient_texname = token;
				continue;
			}

			// diffuse texture
			if ((0 == strncmp(token, "map_Kd", 6)) && isSpace(token[6])) {
				token += 7;
				material.diffuse_texname = token;
				continue;
			}

			// specular texture
			if ((0 == strncmp(token, "map_Ks", 6)) && isSpace(token[6])) {
				token += 7;
				material.specular_texname = token;
				continue;
			}

			// normal texture
			if ((0 == strncmp(token, "map_Ns", 6)) && isSpace(token[6])) {
				token += 7;
				material.normal_texname = token;
				continue;
			}

			// unknown parameter
			const char *_space = strchr(token, ' ');
			if (!_space) {
				_space = strchr(token, '\t');
			}
			if (_space) {
				std::ptrdiff_t len = _space - token;
				std::string key(token, len);
				std::string value = _space + 1;
				material.unknown_parameter.insert(
					std::pair<std::string, std::string>(key, value));
			}
		}
		// flush last material.
		material_map.insert(
			std::pair<std::string, int>(material.name, static_cast<int>(materials.size())));
		materials.push_back(material);

		return err.str();
	}

	std::string MaterialFileReader::operator()(const std::string &matId,
		std::vector<material_t> &materials,
		std::map<std::string, int> &matMap) {
		std::string filepath;

		if (!m_mtlBasePath.empty()) {
			filepath = std::string(m_mtlBasePath) + matId;
		}
		else {
			filepath = matId;
		}

		std::ifstream matIStream(filepath.c_str());
		return LoadMtl(matMap, materials, matIStream);
	}

	std::string LoadObj(std::vector<shape_t> &shapes,
		std::vector<material_t> &materials, // [output]
		const char *filename, const char *mtl_basepath) {

		shapes.clear();

		std::stringstream err;

		std::ifstream ifs(filename);
		if (!ifs) {
			err << "Cannot open file [" << filename << "]" << std::endl;
			return err.str();
		}

		std::string basePath;
		if (mtl_basepath) {
			basePath = mtl_basepath;
		}
		MaterialFileReader matFileReader(basePath);

		return LoadObj(shapes, materials, ifs, matFileReader);
	}

	std::string LoadObj(std::vector<shape_t> &shapes,
		std::vector<material_t> &materials, // [output]
		std::istream &inStream, MaterialReader &readMatFn) {
		std::stringstream err;

		std::vector<float> v;
		std::vector<float> vn;
		std::vector<float> vt;
		std::vector<std::vector<vertex_index> > faceGroup;
		std::string name;

		// material
		std::map<std::string, int> material_map;
		std::map<vertex_index, unsigned int> vertexCache;
		int material = -1;

		shape_t shape;

		int maxchars = 8192;             // Alloc enough size.
		std::vector<char> buf(maxchars); // Alloc enough size.
		while (inStream.peek() != -1) {
			inStream.getline(&buf[0], maxchars);

			std::string linebuf(&buf[0]);

			// Trim newline '\r\n' or '\n'
			if (linebuf.size() > 0) {
				if (linebuf[linebuf.size() - 1] == '\n')
					linebuf.erase(linebuf.size() - 1);
			}
			if (linebuf.size() > 0) {
				if (linebuf[linebuf.size() - 1] == '\r')
					linebuf.erase(linebuf.size() - 1);
			}

			// Skip if empty line.
			if (linebuf.empty()) {
				continue;
			}

			// Skip leading space.
			const char *token = linebuf.c_str();
			token += strspn(token, " \t");

			assert(token);
			if (token[0] == '\0')
				continue; // empty line

			if (token[0] == '#')
				continue; // comment line

						  // vertex
			if (token[0] == 'v' && isSpace((token[1]))) {
				token += 2;
				float x, y, z;
				parseFloat3(x, y, z, token);
				v.push_back(x);
				v.push_back(y);
				v.push_back(z);
				continue;
			}

			// normal
			if (token[0] == 'v' && token[1] == 'n' && isSpace((token[2]))) {
				token += 3;
				float x, y, z;
				parseFloat3(x, y, z, token);
				vn.push_back(x);
				vn.push_back(y);
				vn.push_back(z);
				continue;
			}

			// texcoord
			if (token[0] == 'v' && token[1] == 't' && isSpace((token[2]))) {
				token += 3;
				float x, y;
				parseFloat2(x, y, token);
				vt.push_back(x);
				vt.push_back(y);
				continue;
			}

			// face
			if (token[0] == 'f' && isSpace((token[1]))) {
				token += 2;
				token += strspn(token, " \t");

				std::vector<vertex_index> face;
				while (!isNewLine(token[0])) {
					vertex_index vi =
						parseTriple(token, static_cast<int>(v.size() / 3), static_cast<int>(vn.size() / 3), static_cast<int>(vt.size() / 2));
					face.push_back(vi);
					size_t n = strspn(token, " \t\r");
					token += n;
				}

				faceGroup.push_back(face);

				continue;
			}

			// use mtl
			if ((0 == strncmp(token, "usemtl", 6)) && isSpace((token[6]))) {

				char namebuf[4096];
				token += 7;
#ifdef _MSC_VER
				sscanf_s(token, "%s", namebuf);
#else
				sscanf(token, "%s", namebuf);
#endif

				// Create face group per material.
				bool ret = exportFaceGroupToShape(shape, vertexCache, v, vn, vt,
					faceGroup, material, name, true);
				if (ret) {
					faceGroup.clear();
				}

				if (material_map.find(namebuf) != material_map.end()) {
					material = material_map[namebuf];
				}
				else {
					// { error!! material not found }
					material = -1;
				}

				continue;
			}

			// load mtl
			if ((0 == strncmp(token, "mtllib", 6)) && isSpace((token[6]))) {
				char namebuf[4096];
				token += 7;
#ifdef _MSC_VER
				sscanf_s(token, "%s", namebuf);
#else
				sscanf(token, "%s", namebuf);
#endif

				std::string err_mtl = readMatFn(namebuf, materials, material_map);
				if (!err_mtl.empty()) {
					faceGroup.clear(); // for safety
					return err_mtl;
				}

				continue;
			}

			// group name
			if (token[0] == 'g' && isSpace((token[1]))) {

				// flush previous face group.
				bool ret = exportFaceGroupToShape(shape, vertexCache, v, vn, vt,
					faceGroup, material, name, true);
				if (ret) {
					shapes.push_back(shape);
				}

				shape = shape_t();

				// material = -1;
				faceGroup.clear();

				std::vector<std::string> names;
				while (!isNewLine(token[0])) {
					std::string str = parseString(token);
					names.push_back(str);
					token += strspn(token, " \t\r"); // skip tag
				}

				assert(names.size() > 0);

				// names[0] must be 'g', so skip the 0th element.
				if (names.size() > 1) {
					name = names[1];
				}
				else {
					name = "";
				}

				continue;
			}

			// object name
			if (token[0] == 'o' && isSpace((token[1]))) {

				// flush previous face group.
				bool ret = exportFaceGroupToShape(shape, vertexCache, v, vn, vt,
					faceGroup, material, name, true);
				if (ret) {
					shapes.push_back(shape);
				}

				// material = -1;
				faceGroup.clear();
				shape = shape_t();

				// @todo { multiple object name? }
				char namebuf[4096];
				token += 2;
#ifdef _MSC_VER
				sscanf_s(token, "%s", namebuf);
#else
				sscanf(token, "%s", namebuf);
#endif
				name = std::string(namebuf);

				continue;
			}

			// Ignore unknown command.
		}

		bool ret = exportFaceGroupToShape(shape, vertexCache, v, vn, vt, faceGroup,
			material, name, true);
		if (ret) {
			shapes.push_back(shape);
		}
		faceGroup.clear(); // for safety

		return err.str();
	}
}
