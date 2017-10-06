#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include <GLFW/glfw3.h>

#include "tiny_obj_loader.h"
#include <memory>




string input_filename;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
	input_filename = filename;
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
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "ENVIRONMENTMAP") == 0) {
				loadEnvironment();
				cout << " " << endl;
			}
        }
    }
	fp_in.close();

#ifdef ENABLE_BVH
	double startTime = glfwGetTime();
	// Max number of primitives in BVH node
	bvh_totalNodes = 0;
	int maxPrimsInNode = 5; // TODO : Tweak this magic number for the best performance
	bvh_nodes = ConstructBVHAccel(bvh_totalNodes, tris, maxPrimsInNode);

	cout << "Total time to Consturst a BVH : " << (glfwGetTime() - startTime) << " seconds" << endl;
	cout << "BVH has " << bvh_totalNodes << " nodes totally" << endl;
	cout << endl;

#endif  

	//Process lights
	float sum_area = 0.f;
	for (int i = 0; i < lights.size(); i++) {
		sum_area += lights[i].SurfaceArea;
	}
	float accum_area = 0.f;
	for (int i = 0; i < lights.size(); i++) {
		accum_area += lights[i].SurfaceArea;
		lights[i].selectedProb = accum_area / sum_area;
	}


}

Scene::~Scene() {

#ifdef ENABLE_BVH
	DeconstructBVHAccel(bvh_nodes);

#endif 

	// Free Texture
	for (int i = 0; i < textureMap.size(); i++) {
		textureMap[i].Free();
	}

	// Free normal map
	for (int i = 0; i < normalMap.size(); i++) {
		normalMap[i].Free();
	}

	// Free environment map
	for (int i = 0; i < EnvironmentMap.size(); i++) {
		EnvironmentMap[i].Free();
	}

}


string GetLocalPath() {
	// Get local path
	int len = input_filename.length() - 1;
	while (input_filename[len] != '/')
	{
		len--;
	}
	return input_filename.substr(0, len + 1);
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
		bool isMesh = false;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(line.c_str(), "mesh") == 0) {
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
				isMesh = true;
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
		for(int i = 0; i < 3; i++)
		{	
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);

			//load tranformations
			if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
		}

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);


		//load mesh obj
		if (isMesh) {
			// get obj file name
			utilityCore::safeGetline(fp_in, line);
			string objPath = GetLocalPath() + line;
			cout << "Obj file to open : " << objPath << endl;

			loadObj(objPath, newGeom, newGeom.transform, newGeom.invTranspose);
		}

		// load extra empty lines
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			utilityCore::safeGetline(fp_in, line);
		}

        geoms.push_back(newGeom);

		// Load Lights
		if (find(emitMaterialId.begin(), emitMaterialId.end(), newGeom.materialid) != emitMaterialId.end()) {
			Light newLight;
			newLight.geom = newGeom;
			newLight.geomIdx = id;
			newLight.emittance = materials[newGeom.materialid].emittance * materials[newGeom.materialid].color;
			glm::vec3 scale;
			// Calculate surface area
			switch (newGeom.type)
			{
			case SPHERE:
				scale = newGeom.scale;
				// assume it's a perfect spher, not Ellipsoid
				newLight.SurfaceArea = 4.0f * PI * scale.x * scale.x;
				break;
			case CUBE:
				scale = newGeom.scale;
				newLight.SurfaceArea = 2.0f * (scale.x * scale.y + scale.x * scale.z + scale.y * scale.z);
				break;
			case MESH:
				newLight.SurfaceArea = 0.f;
				for (int i = newGeom.meshTriangleStartIdx; i < newGeom.meshTriangleEndIdx; i++) {
					newLight.SurfaceArea += tris[i].SurfaceArea();
				}
				break;
				// Add more light geom type here
			}
			lights.push_back(newLight);
		}

        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

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
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
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
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

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

        //load static properties
        for (int i = 0; i < 7; i++) {   
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
				if (newMaterial.emittance > FLT_EPSILON) {
					emitMaterialId.push_back(id);
				}
            }
        }

		newMaterial.textureID = -1;
		newMaterial.normalID = -1;

		// load extra map information
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			if (strcmp(tokens[0].c_str(), "textureMap") == 0) {
				
				string texturePath = GetLocalPath() + "tex_nor_maps/" + tokens[1];

				Texture newTexture;
				newTexture.LoadFromFile(texturePath.c_str());

				newMaterial.textureID = textureMap.size();
				textureMap.push_back(newTexture);
			}
			else if (strcmp(tokens[0].c_str(), "normalMap") == 0) {

				string texturePath = GetLocalPath() + "tex_nor_maps/" + tokens[1];

				Texture newTexture;
				newTexture.LoadFromFile(texturePath.c_str());

				newMaterial.normalID = normalMap.size();
				normalMap.push_back(newTexture);
			}
			// Handle other maps here!
			//else if ()
			//{

			//}

			utilityCore::safeGetline(fp_in, line);
		}


        materials.push_back(newMaterial);
        return 1;
    }
}

void Scene::loadObj(string objPath, Geom& newGeom, const glm::mat4& transform, const glm::mat4& invTranspose) {
	std::vector<tinyobj::shape_t> shapes; 
	std::vector<tinyobj::material_t> materials;

	std::string errors = tinyobj::LoadObj(shapes, materials, objPath.c_str());
	std::cout << errors << std::endl;

	if (errors.size() == 0)
	{
		//Read the information from the vector of shape_ts
		newGeom.meshTriangleStartIdx = tris.size();

#ifdef ENABLE_MESHWORLDBOUND
		newGeom.worldBoundIdx = worldBounds.size();
		float min_x = FLT_MAX;
		float min_y = FLT_MAX;
		float min_z = FLT_MAX;

		float max_x = FLT_MIN;
		float max_y = FLT_MIN;
		float max_z = FLT_MIN;
#endif 



		for (unsigned int i = 0; i < shapes.size(); i++)
		{
			std::vector<float> &positions = shapes[i].mesh.positions;
			std::vector<float> &normals = shapes[i].mesh.normals;
			std::vector<float> &uvs = shapes[i].mesh.texcoords;
			std::vector<unsigned int> &indices = shapes[i].mesh.indices;
			for (unsigned int j = 0; j < indices.size(); j += 3)
			{
				glm::vec3 p1 = glm::vec3(transform * glm::vec4(positions[indices[j] * 3], positions[indices[j] * 3 + 1], positions[indices[j] * 3 + 2], 1));
				glm::vec3 p2 = glm::vec3(transform * glm::vec4(positions[indices[j + 1] * 3], positions[indices[j + 1] * 3 + 1], positions[indices[j + 1] * 3 + 2], 1));
				glm::vec3 p3 = glm::vec3(transform * glm::vec4(positions[indices[j + 2] * 3], positions[indices[j + 2] * 3 + 1], positions[indices[j + 2] * 3 + 2], 1));

				Triangle t;
				t.vertices[0] = p1;
				t.vertices[1] = p2;
				t.vertices[2] = p3;

#ifdef ENABLE_MESHWORLDBOUND
				float min_x_temp = glm::min(p1.x, glm::min(p2.x, p3.x));
				float min_y_temp = glm::min(p1.y, glm::min(p2.y, p3.y));
				float min_z_temp = glm::min(p1.z, glm::min(p2.z, p3.z));

				float max_x_temp = glm::max(p1.x, glm::max(p2.x, p3.x));
				float max_y_temp = glm::max(p1.y, glm::max(p2.y, p3.y));
				float max_z_temp = glm::max(p1.z, glm::max(p2.z, p3.z));

				min_x = min_x < min_x_temp ? min_x : min_x_temp;
				min_y = min_y < min_y_temp ? min_y : min_y_temp;
				min_z = min_z < min_z_temp ? min_z : min_z_temp;
				max_x = max_x > max_x_temp ? max_x : max_x_temp;
				max_y = max_y > max_y_temp ? max_y : max_y_temp;
				max_z = max_z > max_z_temp ? max_z : max_z_temp;

#endif
				if (normals.size() > 0)
				{
					glm::vec4 n1 = invTranspose * glm::vec4(normals[indices[j] * 3], normals[indices[j] * 3 + 1], normals[indices[j] * 3 + 2], 0);
					glm::vec4 n2 = invTranspose * glm::vec4(normals[indices[j + 1] * 3], normals[indices[j + 1] * 3 + 1], normals[indices[j + 1] * 3 + 2], 0);
					glm::vec4 n3 = invTranspose * glm::vec4(normals[indices[j + 2] * 3], normals[indices[j + 2] * 3 + 1], normals[indices[j + 2] * 3 + 2], 0);
					t.normals[0] = glm::vec3(n1);
					t.normals[1] = glm::vec3(n2);
					t.normals[2] = glm::vec3(n3);
				}
				if (uvs.size() > 0)
				{
					glm::vec2 t1(uvs[indices[j] * 2], uvs[indices[j] * 2 + 1]);
					glm::vec2 t2(uvs[indices[j + 1] * 2], uvs[indices[j + 1] * 2 + 1]);
					glm::vec2 t3(uvs[indices[j + 2] * 2], uvs[indices[j + 2] * 2 + 1]);
					t.uvs[0] = t1;
					t.uvs[1] = t2;
					t.uvs[2] = t3;
				}
				tris.push_back(t);
			}
		}

		newGeom.meshTriangleEndIdx = tris.size();

#ifdef ENABLE_MESHWORLDBOUND
		worldBounds.push_back(Bounds3f(glm::vec3(min_x, min_y, min_z), 
									   glm::vec3(max_x, max_y, max_z)));
#endif
	}
	else
	{
		std::cout << errors << std::endl;
	}
}


int Scene::loadEnvironment() {
	cout << "Loading Environment Map ..." << endl;

	string line;
	utilityCore::safeGetline(fp_in, line);
	if (!line.empty() && fp_in.good()) {
		string texturePath = GetLocalPath() + "tex_nor_maps/" + line;

		Texture newTexture;

		newTexture.LoadFromFile(texturePath.c_str());

		EnvironmentMap.push_back(newTexture);
	}
	else {
		cout << "ERROR : Loading Environment Map failed" << endl;
	}
}