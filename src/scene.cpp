#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

//#define DEBUGSPHERES 1
//#define DEBUGTRIS

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

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

#ifdef DEBUGSPHERES
	for (int i = 0; i < 5000; i++) {
		Geom newsphere;
		newsphere.type = TRIANGLE;
		newsphere.points[0] = glm::vec3(0, 0, 0);
		newsphere.points[1] = glm::vec3(1, 0, 0);
		newsphere.points[2] = glm::vec3(0, 1, 0);
		newsphere.materialid = 1 + rand() % (materials.size() - 1);
		if (rand() % 1000 < 50) newsphere.materialid = 0; // light
		float x1 = (float)(rand() % 5000) / 2500.0f - 1.0f;
		float x2 = (float)(rand() % 5000) / 2500.0f - 1.0f;
		float x3 = (float)(rand() % 5000) / 2500.0f - 1.0f;
		float xr = (float)(rand() % 5000) / 5000.0f;
		//xr = pow(xr, 0.3333f);
		float scalar = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
		newsphere.translation = xr * 5.0f / scalar * glm::vec3(x1, x2, x3) + glm::vec3(0, 5, 0);
		
		float rr = (float)(rand() % 5000) / 5000.0f;
		rr = rr * 0.4f + 0.1f;

		float rrx = (float)(rand() % 5000) / 5000.0f * 360.0f;
		float rry = (float)(rand() % 5000) / 5000.0f * 360.0f;
		float rrz = (float)(rand() % 5000) / 5000.0f * 360.0f;
		newsphere.rotation = glm::vec3(rrx, rry, rrz);
		//newsphere.rotation = glm::vec3(0);

		newsphere.scale = glm::vec3(rr);
		newsphere.transform = utilityCore::buildTransformationMatrix(newsphere.translation, newsphere.rotation, newsphere.scale);
		newsphere.inverseTransform = glm::inverse(newsphere.transform);
		newsphere.invTranspose = glm::inverseTranspose(newsphere.transform);

		geoms.push_back(newsphere);
	}
#endif // DEBUGSPHERES

#ifdef DEBUGTRIS
	Geom tri1;
	tri1.type = TRIANGLE;
	tri1.points[0] = glm::vec3(-5, 0, -1);
	tri1.points[1] = glm::vec3(5, 0, -1);
	tri1.points[2] = glm::vec3(-5, 10, -1);
	tri1.translation = glm::vec3(0);
	tri1.rotation = glm::vec3(0);
	tri1.scale = glm::vec3(1);
	tri1.transform = utilityCore::buildTransformationMatrix(tri1.translation, tri1.rotation, tri1.scale);
	tri1.inverseTransform = glm::inverse(tri1.transform);
	tri1.invTranspose = glm::inverseTranspose(tri1.transform);
	tri1.materialid = 1;

	Geom tri2;
	tri2.type = TRIANGLE;
	tri2.points[0] = glm::vec3(-5, 0, -2);
	tri2.points[1] = glm::vec3(5, 10, -2);
	tri2.points[2] = glm::vec3(-5, 10, -2);
	tri2.translation = glm::vec3(0);
	tri2.rotation = glm::vec3(0);
	tri2.scale = glm::vec3(1);
	tri2.transform = utilityCore::buildTransformationMatrix(tri2.translation, tri2.rotation, tri2.scale);
	tri2.inverseTransform = glm::inverse(tri2.transform);
	tri2.invTranspose = glm::inverseTranspose(tri2.transform);
	tri2.materialid = 2;

	Geom tri3;
	tri3.type = TRIANGLE;
	tri3.points[0] = glm::vec3(0, 0, 5);
	tri3.points[1] = glm::vec3(0, 0, -5);
	tri3.points[2] = glm::vec3(0, 10, -5);
	tri3.translation = glm::vec3(0);
	tri3.rotation = glm::vec3(0, 0, 45);
	tri3.scale = glm::vec3(1);
	tri3.transform = utilityCore::buildTransformationMatrix(tri3.translation, tri3.rotation, tri3.scale);
	tri3.inverseTransform = glm::inverse(tri3.transform);
	tri3.invTranspose = glm::inverseTranspose(tri3.transform);
	tri3.materialid = 3;

	geoms.push_back(tri1);
	geoms.push_back(tri2);
	geoms.push_back(tri3);

#endif // DEBUGTRIS


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
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}

			utilityCore::safeGetline(fp_in, line);
		}

		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);


		if (newGeom.type != MESH) {
			geoms.push_back(newGeom);
		}
		else {
			// load the mesh with tinyobjloader
			// for each triangle, apply transform and push to geom list
			string objFile;
			utilityCore::safeGetline(fp_in, line);
			objFile = line;
			cout << "OBJ: " << objFile << "\n";
			string error;
			std::vector<tinyobj::shape_t> shapes;
			std::vector<tinyobj::material_t> mats; // will be discarded
			tinyobj::attrib_t attr;
			bool success = tinyobj::LoadObj(&attr, &shapes, &mats, &error, objFile.c_str());
			if (!success) return -1; // ?
			if (!error.empty()) {
				cout << "Error loading obj! " << error << "\n";
				//return -1;
			}

			// decompose loaded mesh into tris, set attributes to everything read before
			// for separate shapes in obj
			for (int i = 0; i < shapes.size(); i++) {
				
				// for each triangle
				for (int j = 0; j < shapes[i].mesh.indices.size() / 3; j++) {
					
					Geom newTri;
					newTri.type = TRIANGLE;
					newTri.materialid = newGeom.materialid;
					newTri.transform = newGeom.transform;
					newTri.invTranspose = newGeom.invTranspose;
					newTri.inverseTransform = newGeom.inverseTransform;
					newTri.translation = newGeom.translation;
					newTri.rotation = newGeom.rotation;
					newTri.scale = newGeom.scale;

					// for each vertex
					for (int k = 0; k < 3; k++) {
						int triIdx = shapes[i].mesh.indices[3 * j + k].vertex_index;
						newTri.points[k] = glm::vec3(
							attr.vertices[3 * triIdx],
							attr.vertices[3 * triIdx + 1],
							attr.vertices[3 * triIdx + 2]
							);		
					}
					geoms.push_back(newTri);

					//cout << "made triangle with pts: \n\t" << newTri.points[0].x << ", " << newTri.points[0].y << ", " << newTri.points[0].z << "\n\t"
					//	<< newTri.points[1].x << ", " << newTri.points[1].y << ", " << newTri.points[1].z << "\n\t"
					//	<< newTri.points[2].x << ", " << newTri.points[2].y << ", " << newTri.points[2].z << "\n";
				}
			}

		}

        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

	camera.focusDistance = 5.0f;
	camera.lensRadius = 0.1f;

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

