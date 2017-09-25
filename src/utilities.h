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

	//Functions obtained from globals file of my Monte Carlo PathTracer made in CIS 561
	//A bunch of useful functions for path tracing, all have self explanatory names
	//extern inline bool IsBlack(const Color3f& c)
	//{
	//	return (c.r == 0.f && c.g == 0.f && c.b == 0.f);
	//}

	//extern inline float AbsDot(const Vector3f& a, const Vector3f& b)
	//{
	//	return glm::abs(glm::dot(a, b));
	//}

	//extern inline bool SameHemisphere(const Vector3f &w, const Vector3f &wp)
	//{
	//	return w.z * wp.z > 0;
	//}

	//extern inline float CosTheta(const Vector3f &w) { return w.z; }
	//extern inline float Cos2Theta(const Vector3f &w) { return w.z * w.z; }
	//extern inline float AbsCosTheta(const Vector3f &w) { return std::abs(w.z); }
	//extern inline float Sin2Theta(const Vector3f &w)
	//{
	//	return std::max((float)0, (float)1 - Cos2Theta(w));
	//}

	//extern inline float SinTheta(const Vector3f &w)
	//{ 
	//	return std::sqrt(Sin2Theta(w)); 
	//}

	//extern inline float TanTheta(const Vector3f &w)
	//{ 
	//	return SinTheta(w) / CosTheta(w);
	//}

	//extern inline float Tan2Theta(const Vector3f &w) {
	//	return Sin2Theta(w) / Cos2Theta(w);
	//}

	//extern inline float CosPhi(const Vector3f &w)
	//{
	//	float sinTheta = SinTheta(w);
	//	return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
	//}

	//extern inline float SinPhi(const Vector3f &w)
	//{
	//	float sinTheta = SinTheta(w);
	//	return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
	//}

	//extern inline float Cos2Phi(const Vector3f &w)
	//{ 
	//	return CosPhi(w) * CosPhi(w); 
	//}

	//extern inline float Sin2Phi(const Vector3f &w)
	//{ 
	//	return SinPhi(w) * SinPhi(w); 
	//}
	//
	//extern inline bool Refract(const Vector3f &wi, const Normal3f &n, float eta, Vector3f *wt)
	//{
	//	 Compute cos theta using Snell's law
	//	float cosThetaI = glm::dot(n, wi);
	//	float sin2ThetaI = std::max(float(0), float(1 - cosThetaI * cosThetaI));
	//	float sin2ThetaT = eta * eta * sin2ThetaI;

	//	 Handle total internal reflection for transmission
	//	if (sin2ThetaT >= 1) return false;
	//	float cosThetaT = std::sqrt(1 - sin2ThetaT);
	//	*wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Vector3f(n);
	//	return true;
	//}
	//
	//extern inline Normal3f Faceforward(const Normal3f &n, const Vector3f &v)
	//{
	//	return (glm::dot(n, v) < 0.f) ? -n : n;
	//}
}
