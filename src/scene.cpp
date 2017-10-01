#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

// External Libaray
// tinyobjloader
// https://github.com/syoyo/tinyobjloader
#include "tiny_obj_loader.h"

Scene::Scene(string filename, float aTime) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
	
	aniTime = aTime;

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

inline float max4(float a, float b, float c, float d) {
	return max(a, max(b, max(c, d)));
}

inline float min4(float a, float b, float c, float d) {
	return min(a, min(b, min(c, d)));
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
			} else if (strcmp(line.c_str(), "mesh") == 0) {
				cout << "Creating new triangle mesh..." << endl;
				newGeom.type = TRIS;
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
			} else if (strcmp(tokens[0].c_str(), "MT") == 0) {
				newGeom.mt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			} else if (strcmp(tokens[0].c_str(), "MR") == 0) {
				newGeom.mr = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			
			else if (strcmp(tokens[0].c_str(), "FPATH") == 0) {
				std::string fpath;
				for (size_t i = 1; i < tokens.size(); ++i) {
					fpath += tokens[i] + " ";
				}

				cout << "Load Mesh from " << fpath << endl;
				
				using namespace tinyobj;
				typedef std::vector<real_t>  mvec;
				typedef std::vector<index_t> midx;

				float x_max, y_max, z_max;
				float x_min, y_min, z_min;

				x_min = INT_MAX;
				y_min = INT_MAX;
				z_min = INT_MAX;
				x_max = INT_MIN;
				y_max = INT_MIN;
				z_max = INT_MIN;

				attrib_t a;
				std::vector<shape_t> s;
				std::vector<material_t> m;
				
				// Load *.obj file
				LoadObj(&a, &s, &m, NULL, fpath.c_str());

				mvec &vtx = a.vertices;
				mvec &n	  = a.normals;

				int ntris = 0;
				for (shape_t si : s) {
					ntris += (si.mesh.indices.size() / 3);
				}

				for (shape_t si : s) 
				{
					midx &ids = si.mesh.indices;
					for (size_t i = 0; i < ids.size() / 3; i++) {
						Tri t;
						t.idx = id;
						int i0 = ids[3 * i + 0].vertex_index;
						int i1 = ids[3 * i + 1].vertex_index;
						int i2 = ids[3 * i + 2].vertex_index;
						int n0 = ids[3 * i + 0].normal_index;
						int n1 = ids[3 * i + 1].normal_index;
						int n2 = ids[3 * i + 2].normal_index;
						t.position[0] = glm::vec3(vtx[3 * i0 + 0], vtx[3 * i0 + 1], vtx[3 * i0 + 2]);
						t.position[1] = glm::vec3(vtx[3 * i1 + 0], vtx[3 * i1 + 1], vtx[3 * i1 + 2]);
						t.position[2] = glm::vec3(vtx[3 * i2 + 0], vtx[3 * i2 + 1], vtx[3 * i2 + 2]);
						t.normal[0] = glm::vec3(n[3 * n0 + 0], n[3 * n0 + 1], n[3 * n0 + 2]);
						t.normal[1] = glm::vec3(n[3 * n1 + 0], n[3 * n1 + 1], n[3 * n1 + 2]);
						t.normal[2] = glm::vec3(n[3 * n2 + 0], n[3 * n2 + 1], n[3 * n2 + 2]);

						x_max = max4(t.position[0].x, t.position[1].x, t.position[2].x, x_max);
						y_max = max4(t.position[0].y, t.position[1].y, t.position[2].y, y_max);
						z_max = max4(t.position[0].z, t.position[1].z, t.position[2].z, z_max);
						x_min = min4(t.position[0].x, t.position[1].x, t.position[2].x, x_min);
						y_min = min4(t.position[0].y, t.position[1].y, t.position[2].y, y_min);
						z_min = min4(t.position[0].z, t.position[1].z, t.position[2].z, z_min);

						tris.push_back(t);
					}
				}

				newGeom.a = glm::vec3(x_max, y_max, z_max);
				newGeom.b = glm::vec3(x_min, y_min, z_min);
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
		else if (strcmp(tokens[0].c_str(), "AP") == 0) {
			camera.ap = stof(tokens[1]);
		}
		else if (strcmp(tokens[0].c_str(), "F") == 0) {
			camera.f = stof(tokens[1]);
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
