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
#define INVPI             0.31830988618379067154f
#define INV2PI            0.15915494309189533577f
#define INV4PI            0.07957747154594766788f
#define PI_OVER_2         1.57079632679489661923f
#define PI_OVER_4         0.78539816339744830961f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define SQRT_OF_TWO       1.41421356237309504880f
#define EPSILON           0.00001f
#define OneMinusEpsilon   0.99999994f

// We're going to create some type aliases to
// give our code a little more context so it's
// easier to interpret at a glance.
// For example, we're going to say that our
// custom types Color3f, Point3f, and Vector3f are
// all aliases of glm::vec3. Since we'll use them in
// different contexts, we want the human-readable distinction.
typedef float Float;
typedef glm::vec3 Color3f;
typedef glm::vec3 Point3f;
typedef glm::vec3 Normal3f;
typedef glm::vec2 Point2f;
typedef glm::ivec2 Point2i;
typedef glm::ivec3 Point3i;
typedef glm::vec3 Vector3f;
typedef glm::vec2 Vector2f;
typedef glm::ivec2 Vector2i;
typedef glm::mat4 Matrix4x4;
typedef glm::mat3 Matrix3x3;


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
