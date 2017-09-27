#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include "tinyobj\tiny_obj_loader.h"

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
		std::vector<Geom> newGeoms;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
				Geom geo;
                geo.type = SPHERE;
				geo.nextIdxOff = 0;
				newGeoms.push_back(geo);
            } else if (strcmp(tokens[0].c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
				Geom geo;
                geo.type = CUBE;
				geo.nextIdxOff = 0;
				newGeoms.push_back(geo);
            }
			else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
				cout << "Creating new mesh..." << endl;
				tinyobj::attrib_t attrib;
				std::vector<tinyobj::shape_t> shapes; 
				std::vector<tinyobj::material_t> materials;
				std::string err;
				bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, tokens[1].c_str());
				if (!err.empty()) {
					cout << err << endl;
				}
				if (!ret) {
					cout << "Failed to load " << tokens[1].c_str() << endl;
				}

				printf("# of vertices  = %d\n", (int)(attrib.vertices.size()) / 3);
				printf("# of normals   = %d\n", (int)(attrib.normals.size()) / 3);
				printf("# of texcoords = %d\n", (int)(attrib.texcoords.size()) / 2);
				printf("# of materials = %d\n", (int)materials.size());
				printf("# of shapes    = %d\n", (int)shapes.size());

				float bmin[3];
				float bmax[3];
				bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
				bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

				for (unsigned int s = 0; s < shapes.size(); ++s) {
					for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++) {
						Geom geo;
						geo.type = TRIANGLE;
						geo.nextIdxOff = 0;

						tinyobj::index_t idx0 = shapes[s].mesh.indices[3 * f + 0];
						tinyobj::index_t idx1 = shapes[s].mesh.indices[3 * f + 1];
						tinyobj::index_t idx2 = shapes[s].mesh.indices[3 * f + 2];

						// positions
						float v[3][3];
						for (int k = 0; k < 3; k++) {
							int f0 = idx0.vertex_index;
							int f1 = idx1.vertex_index;
							int f2 = idx2.vertex_index;
							assert(f0 >= 0);
							assert(f1 >= 0);
							assert(f2 >= 0);

							v[0][k] = attrib.vertices[3 * f0 + k];
							v[1][k] = attrib.vertices[3 * f1 + k];
							v[2][k] = attrib.vertices[3 * f2 + k];
							bmin[k] = std::min(v[0][k], bmin[k]);
							bmin[k] = std::min(v[1][k], bmin[k]);
							bmin[k] = std::min(v[2][k], bmin[k]);
							bmax[k] = std::max(v[0][k], bmax[k]);
							bmax[k] = std::max(v[1][k], bmax[k]);
							bmax[k] = std::max(v[2][k], bmax[k]);
						}
						geo.pos[0] = glm::vec3(v[0][0], v[0][1], v[0][2]);
						geo.pos[1] = glm::vec3(v[1][0], v[1][1], v[1][2]);
						geo.pos[2] = glm::vec3(v[2][0], v[2][1], v[2][2]);

						// normals
						float n[3][3];
						if (attrib.normals.size() > 0) {
							int f0 = idx0.normal_index;
							int f1 = idx1.normal_index;
							int f2 = idx2.normal_index;
							assert(f0 >= 0);
							assert(f1 >= 0);
							assert(f2 >= 0);
							for (int k = 0; k < 3; k++) {
								n[0][k] = attrib.normals[3 * f0 + k];
								n[1][k] = attrib.normals[3 * f1 + k];
								n[2][k] = attrib.normals[3 * f2 + k];
							}
						} else {
							// compute geometric normal
							CalcNormal(n[0], v[0], v[1], v[2]);
							n[1][0] = n[0][0];
							n[1][1] = n[0][1];
							n[1][2] = n[0][2];
							n[2][0] = n[0][0];
							n[2][1] = n[0][1];
							n[2][2] = n[0][2];
						}
						geo.norm[0] = glm::vec3(n[0][0], n[0][1], n[0][2]);
						geo.norm[1] = glm::vec3(n[1][0], n[1][1], n[1][2]);
						geo.norm[2] = glm::vec3(n[2][0], n[2][1], n[2][2]);

						newGeoms.push_back(geo);
					}
				}
				Geom geo;
				geo.type = MESH;
				geo.nextIdxOff = newGeoms.size();
				geo.bound[0] = glm::vec3(bmin[0], bmin[1], bmin[2]);
				geo.bound[1] = glm::vec3(bmax[0], bmax[1], bmax[2]);
				geoms.push_back(geo);
			}
        }

		int num_geo = newGeoms.size();

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
			for (int i = 0; i < num_geo; ++i) {
				Geom &geo = newGeoms[i];
				geo.materialid = atoi(tokens[1].c_str());
			}
            cout << "Connecting Geom " << objectid << " to Material " << tokens[1].c_str() << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				for (int i = 0; i < num_geo; ++i) {
					Geom &geo = newGeoms[i];
					geo.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				}
                
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				for (int i = 0; i < num_geo; ++i) {
					Geom &geo = newGeoms[i];
					geo.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				}
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				for (int i = 0; i < num_geo; ++i) {
					Geom &geo = newGeoms[i];
					geo.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				}
            }

            utilityCore::safeGetline(fp_in, line);
        }

		for (int i = 0; i < num_geo; ++i) {
			Geom &geo = newGeoms[i];
			geo.transform = utilityCore::buildTransformationMatrix(geo.translation, geo.rotation, geo.scale);
			geo.inverseTransform = glm::inverse(geo.transform);
			geo.invTranspose = glm::inverseTranspose(geo.transform);

			geoms.push_back(geo);
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

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
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
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
