#pragma once

#include "sampling.h"

__host__ __device__ float HenyeyGreensteinPhaseFunction(const Vector3f &wo, const Vector3f & wi, const float& g) 
{ 
	//implementing Henyey-Greenstein Anisotropic scattering
	//Phase functions are used as pdfs for the scattered direction
	float dot = glm::dot(wo, wi);
	float cosTheta = glm::cos(dot);

	float numerator = 1.0f - g*g;
	float denominator = 4 * PI*(glm::pow((1 + g*g - 2 * g*cosTheta), 1.5f));
	return numerator / denominator;
}

__host__ __device__ float smootherstep(float& x)
{
	return 6.0f*glm::pow(x, 5) - 15.0f*glm::pow(x,4) + 10.0f*glm::pow(x,3);
}

__host__ __device__ float sampleScatterDistance(float& samplePoint, float& scatteringCoefficient)
{
	//scattering coefficient scales up the distance by a good deal
	//-logf of samplePoint would be a value b/w 2 and 0
	if (samplePoint < 0.01f)
	{
		//logf(0) = undefined
		samplePoint = 0.01f;
	}
	return -logf(samplePoint) / scatteringCoefficient;
}

__host__ __device__ Vector3f squareToConcentricDiscExponentialFallOffDistribution(Point2f& sample,
												const float& exponentialDistribution_RadiusCoefficient)
{
	glm::vec2 sampleOffset = 2.0f*sample - glm::vec2(1, 1);
	if (sampleOffset.x == 0 && sampleOffset.y == 0)
	{
		return Point3f(0.0f, 0.0f, 0.0f);
	}

	float theta, r, exponentialDistribution_Radius;
	if (std::abs(sampleOffset.x) > std::abs(sampleOffset.y))
	{
		r = sampleOffset.x;
		theta = (PI / 4.0f) * (sampleOffset.y / sampleOffset.x);
	}
	else
	{
		r = sampleOffset.y;
		theta = (PI / 2.0f) - (PI / 4.0f) * (sampleOffset.x / sampleOffset.y);
	}

	exponentialDistribution_Radius = r*exponentialDistribution_RadiusCoefficient;

	//y is up
	return exponentialDistribution_Radius*Point3f(std::cos(theta), 0.0f, std::sin(theta));
}

__host__ __device__ void generateScatteredDirectionandPDF(const Point2f &sample, float& thetaMin,
														  float& pdf, Vector3f& wi, const Vector3f& wo)
{
	//translate the thetaMin into a g value
	float normalizedthetaMin = thetaMin / 180.0f;
	float g = smootherstep(normalizedthetaMin);

	Vector3f scatteredDirection = squareToSphereCapUniform(sample, thetaMin);
	wi = scatteredDirection;
	pdf = HenyeyGreensteinPhaseFunction(wo, wi, g);
}

__host__ __device__ Vector3f axisAngletoEuler(Vector3f& vec, float& angle)
{
	float s = glm::sin(angle);
	float c = glm::cos(angle);
	float t = 1.0f - c;

	Vector3f eulerangles; //as attitude, heading, bank which is x,y,z

	if ((vec.x*vec.y*t + vec.z*s) > 0.998) 
	{ 
		// north pole singularity detected
		eulerangles.x = 2.0f * std::atan2(vec.x*glm::sin(angle*0.5f), glm::cos(angle*0.5f));
		eulerangles.y = PI * 0.5f;
		eulerangles.z = 0.0f;
		return eulerangles;
	}
	if ((vec.x*vec.y*t + vec.z*s) < -0.998)
	{ 
		// south pole singularity detected
		eulerangles.x = -2.0f * std::atan2(vec.x*glm::sin(angle*0.5f), glm::cos(angle*0.5f));
		eulerangles.y = -PI * 0.5f;
		eulerangles.z = 0.0f;
		return eulerangles;
	}
	eulerangles.x = std::atan2(vec.y * s - vec.x * vec.z * t, 1 - (vec.y*vec.y + vec.z*vec.z) * t);
	eulerangles.y = std::asin(vec.x * vec.y * t + vec.z * s);
	eulerangles.z = std::atan2(vec.x * s - vec.y * vec.z * t, 1 - (vec.x*vec.x + vec.z*vec.z) * t);
	return eulerangles;
}

__host__ __device__ Vector3f generateDisplacedOrigin(Vector3f& normal, Vector3f& intersectionPoint,
													 Vector2f& sample, float& sampledDistance)
{
	//sample the squareToConcentricDiscExponentialFallOffDistribution function to generate a sample in a radial disc 
	//with the ditribution of that exponential function and then transform that to world space.
	float expDistRadialCoeff = 1 / (1 + sampledDistance);
	Point3f discPointExpDistribution = squareToConcentricDiscExponentialFallOffDistribution(sample, expDistRadialCoeff);

	Vector3f local_up = Vector3f(0, 1, 0);
	Vector3f world_normal = glm::normalize(normal);

	Vector3f crossproduct = glm::cross(local_up, world_normal);
	float cosRotAngle = glm::dot(local_up, world_normal); //because both vetors have length == 1
	float rotAngle = glm::acos(cosRotAngle);
	//you now have axis angle form convert to euler angles

	Vector3f eulerangles = axisAngletoEuler(crossproduct, rotAngle);

	Vector3f rotation = eulerangles;
	Vector3f translation = intersectionPoint;
	Vector3f scale = Vector3f(1.0f, 1.0f, 1.0f);
	glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);

	Vector3f newRayOrigin = glm::vec3(transform*glm::vec4(discPointExpDistribution, 1.0f));

	//move newRayOrigin into the object
	newRayOrigin += -glm::normalize(normal)*sampledDistance;

	return newRayOrigin;
}

__host__ __device__ Color3f f_Subsurface(Material &m, Vector3f& wo, Vector3f& wi)
{
	//TODO
	return m.color*INVPI;
}

__host__ __device__ float pdf_Subsurface(Vector3f& wo, Vector3f& wi, float& thetaMin)
{
	float normalizedthetaMin = thetaMin / 180.0f;
	float g = smootherstep(normalizedthetaMin);

	float pdf = HenyeyGreensteinPhaseFunction(wo, wi, g);
	return pdf;
}

__host__ __device__ Color3f sample_f_Subsurface(Vector3f& wo, Vector3f& sample,
												Vector3f& normal, Vector3f& intersectionPoint,
												float& scatteringCoefficient,
												Vector3f& wi, float& pdf)
{
	//TODO
	//generate a ray whose origin is a point on a disc(with exponential distribution) generated around the surfaceNormal
	//and shifted inside the object by some amount
	//and this ray has a direction determined by the generateScatteredDirectionandPDF
	//carry out an intersection test with this new ray that originates from within the object
	//determine the sampleDistance
	//if sampleDistance<t do regluar material stuff
	//ie set the pdf back to zero and and color to zero and sample another bxdf
	//else shift the exit point to the surface of the object, ie make sampledDistance = t
	//set the color to a scaled down value colorised by the material and set the pdf and wi

	Ray ray;
	Vector2f sample2f = Vector2f(sample2f[0], sample2f[1]);
	float sampledDistance = sampleScatterDistance(sample[2], float& scatteringCoefficient);
	ray.origin = generateDisplacedOrigin(normal, intersectionPoint, sample2f, sampledDistance);
	ray.direction = ;

	computeIntersectionsWithSelectedObject(Ray& ray, Geom& geom, ShadeableIntersection& intersection);

	return Color3f(0.0f);
}