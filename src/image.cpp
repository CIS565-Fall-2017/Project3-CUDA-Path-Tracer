#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <stb_image.h>

#include "image.h"

void Texture::Load(const std::string & path)
{
	int channels = 0;
	float * rawPixels = stbi_loadf(path.c_str(), &this->width, &this->height, &channels, 3);

	if (channels == 3 || channels == 4)
	{
		glm::vec3 correction = glm::vec3(gamma);
		this->pixels = new glm::vec3[width * height];

		for (int i = 0; i < width * height; i++)
		{
			glm::vec3 color;
			color.x = rawPixels[i * channels];
			color.y = rawPixels[i * channels + 1];
			color.z = rawPixels[i * channels + 2];
			
			this->pixels[i] = glm::pow(color, correction);
		}

		std::cout << "Loaded texture \"" << path << "\" [" << width << "x" << height << "|" << channels << "]" << std::endl;
	}
	else
	{
		std::cerr << "Error loading texture " << path << std::endl;
	}

	stbi_image_free(rawPixels);
}

Texture::Texture(const std::string & filename, float gamma) : gamma(gamma)
{
	this->Load(filename);
}

Texture::Texture(int x, int y) :
        width(x),
        height(y),
        pixels(new glm::vec3[x * y]),
		gamma(1.f) {
}

Texture::~Texture() {
    delete pixels;
}

void Texture::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < width && y < height);
    pixels[(y * width) + x] = pixel;
}

void Texture::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) { 
            int i = y * width + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), width, height, 3, bytes, width * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Texture::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), width, height, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

float Texture::GetGamma()
{
	return gamma;
}

int Texture::GetWidth()
{
	return width;
}

int Texture::GetHeight()
{
	return height;
}

int Texture::GetBytes()
{
	return width * height * 3;
}

glm::vec3 * Texture::GetData()
{
	return pixels;
}
