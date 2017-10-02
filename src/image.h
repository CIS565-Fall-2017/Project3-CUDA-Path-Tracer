#pragma once

#include <glm/glm.hpp>

using namespace std;

class Texture {
private:
    int width;
    int height;
	float gamma;
	bool normalize;
    glm::vec3 *pixels;

	void Load(const std::string& filename);

public:
	Texture(const std::string& filename, float gamma, bool normalize = false);
    Texture(int x, int y);
    ~Texture();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);

	float GetGamma();
	int GetWidth();
	int GetHeight();
	int GetBytes();
	glm::vec3 * GetData();
};
