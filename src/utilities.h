#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define INVPI             0.31830988618379067154
#define INV2PI            0.15915494309189533577
#define INV4PI            0.07957747154594766788
#define PI_OVER_2           1.57079632679489661923
#define PI_OVER_4           0.78539816339744830961
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define SQRT_OF_TWO       1.41421356237309504880f
#define EPSILON           0.00001f

namespace utilityCore 
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
