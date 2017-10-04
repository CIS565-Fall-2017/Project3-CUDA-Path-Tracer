#include <iostream>
#include "scene.h"
#include "tiny_obj_loader.h"
#include <cstring>
#include <stb_image.h>
#include <stb_image_write.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

//DELET THIS
#include <direct.h>
#define GetCurrentDir _getcwd

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
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "ENVIRONMENT") == 0) {
				loadEnvironment();
				cout << " " << endl;
				cout << "Ooh.. environment map, fancy!" << endl;
			}
        }
    }
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
		bool loadMesh = false;
		string meshFile;

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
			else if (strcmp(line.c_str(), "plane") == 0) {
				cout << "Creating new plane..." << endl;
				newGeom.type = PLANE;
			}
			else if (strcmp(line.c_str(), "mesh") == 0) {
				cout <<"Found mesh..." << endl;
				utilityCore::safeGetline(fp_in, line);
				meshFile = line;
				loadMesh = true;
				newGeom.type = TRIANGLE;
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

		if (loadMesh) {
			loadOBJ(newGeom, meshFile);
		}
		else {
			geoms.push_back(newGeom);
		}

        return 1;
    }
}

void Scene::loadOBJ(Geom& base_tri, string& filename)
{
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials; // will be discarded
	tinyobj::attrib_t attributes;
	string errors;
	tinyobj::LoadObj(&attributes, &shapes, &materials, &errors, filename.c_str());

	if (!errors.empty()) {
		printf("Error loading obj in loadOBJ because: %s\n", errors.c_str());
	}

	if (errors.size() == 0) {
		for (unsigned int i = 0; i < shapes.size(); i++)
		{
			std::vector<float> &positions                = attributes.vertices;
			std::vector<float> &normals                  = attributes.normals;
			std::vector<float> &uvs                      = attributes.texcoords;
			const std::vector<tinyobj::index_t> &indices = shapes[i].mesh.indices;

			for (unsigned int j = 0; j < indices.size() / 3; j ++)
			{
				Geom tri;
				tri.type = TRIANGLE;
				tri.materialid       = base_tri.materialid;
				tri.translation      = base_tri.translation;
				tri.rotation         = base_tri.rotation;
				tri.scale            = base_tri.scale;
				tri.transform        = base_tri.transform;
				tri.invTranspose     = base_tri.invTranspose;
				tri.inverseTransform = base_tri.inverseTransform;

				int idx_j0 = indices[3 * j + 0].vertex_index;
				int idx_j1 = indices[3 * j + 1].vertex_index;
				int idx_j2 = indices[3 * j + 2].vertex_index;
				glm::vec3 p0(positions[idx_j0 * 3], positions[idx_j0 * 3 + 1], positions[idx_j0 * 3 + 2]);
				glm::vec3 p1(positions[idx_j1 * 3], positions[idx_j1 * 3 + 1], positions[idx_j1 * 3 + 2]);
				glm::vec3 p2(positions[idx_j2 * 3], positions[idx_j2 * 3 + 1], positions[idx_j2 * 3 + 2]);

				tri.positions[0] = p0;
				tri.positions[1] = p1;
				tri.positions[2] = p2;

				idx_j0 = indices[3 * j + 0].normal_index;
				idx_j1 = indices[3 * j + 1].normal_index;
				idx_j2 = indices[3 * j + 2].normal_index;
				//Get Normals Indices
				if (normals.size() > 0) //Checking if Normals defined.
				{
					glm::vec3 n1(normals[idx_j0 * 3], normals[idx_j0 * 3 + 1], normals[idx_j0 * 3 + 2]);
					glm::vec3 n2(normals[idx_j1 * 3], normals[idx_j1 * 3 + 1], normals[idx_j1 * 3 + 2]);
					glm::vec3 n3(normals[idx_j2 * 3], normals[idx_j2 * 3 + 1], normals[idx_j2 * 3 + 2]);
					tri.normals[0] = n1;
					tri.normals[1] = n2;
					tri.normals[2] = n3;
				}

				idx_j0 = indices[3 * j + 0].texcoord_index;
				idx_j1 = indices[3 * j + 1].texcoord_index;
				idx_j2 = indices[3 * j + 2].texcoord_index;
				if (uvs.size() > 0) //Checking if UVs defined.
				{
					glm::vec2 t1(uvs[idx_j0 * 2], uvs[idx_j0 * 2 + 1]);
					glm::vec2 t2(uvs[idx_j1 * 2], uvs[idx_j1 * 2 + 1]);
					glm::vec2 t3(uvs[idx_j2 * 2], uvs[idx_j2 * 2 + 1]);
					tri.uvs[0] = t1;
					tri.uvs[1] = t2;
					tri.uvs[2] = t3;
				}

				geoms.push_back(tri);
			}
		}
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
	}
	else {
		cout << "Loading Material " << id << "..." << endl;
		Material newMaterial;

		//load static properties
		for (int i = 0; i < 8; i++) {
			string line;
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0) {
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
				newMaterial.specular.exponent = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specular.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
				newMaterial.emittance = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "BSDF") == 0) {
				newMaterial.bsdf = atoi(tokens[1].c_str());
				if (newMaterial.bsdf == -1) { light_count++; }
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}

int Scene::loadEnvironment() {
	//To Be Filled
	char *name = (char*) malloc(sizeof(char) * FILENAME_MAX);
	int dim_x;   //Width  of Image / Scanline  Size
	int dim_y;   //Height of Image / Scanline Count
	int dim_bpp; //Bytes Per Pixel /    Pixel Depth
	
	for (int i = 0; i < 2; i++) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "FILENAME") == 0) {
			strcpy(name, tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DIMENSIONS") == 0) {
			dim_x = atoi(tokens[1].c_str());
			dim_y = atoi(tokens[2].c_str());
			dim_bpp = atoi(tokens[3].c_str());
		}
	}
	environment_dims = glm::ivec3(dim_x, dim_y, dim_bpp);
	environment = stbi_load(name, &dim_x, &dim_y, &dim_bpp, 0);

	if (environment == NULL) {
		printf("STBI Image Loading failed: %s\n", stbi_failure_reason());
		return -1;
	}

	/**
	printf("First 10 pixels of texture: \n");
	for (int x = 5*dim_bpp; x < 10*dim_bpp; x+= dim_bpp) {
	   unsigned char* index = (environment + x);
	   printf("Pixel is: (%d, %d, %d, %d)  ", *(index + 0), *(index + 1), *(index + 2), *(index + 3));
	}
	**/
	
	return 1;
}

