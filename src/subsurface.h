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

__host__ __device__ float sampleScatterDistance(float& samplePoint, const float& scatteringCoefficient)
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
	//wi = wo;
	//printf("%f %f %f \n", wi.x, wi.y, wi.z);
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

__host__ __device__ glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) 
{
	glm::mat4 translationMat = glm::mat4();
	translationMat[3] = glm::vec4(translation, 1.0f);
	
	glm::mat4 rotationMat = glm::mat4();
	glm::mat4 rx = glm::mat4();
	glm::mat4 ry = glm::mat4();
	glm::mat4 rz = glm::mat4();

	rx[1][1] = glm::cos(rotation.x);
	rx[2][1] = -glm::sin(rotation.x);
	rx[1][2] = glm::sin(rotation.x);
	rx[2][2] = glm::cos(rotation.x);

	ry[0][0] = glm::cos(rotation.y);
	ry[2][0] = glm::sin(rotation.y);
	ry[0][2] = -glm::sin(rotation.y);
	ry[2][2] = glm::cos(rotation.y);

	rz[0][0] = glm::cos(rotation.z);
	rz[1][0] = -glm::sin(rotation.z);
	rz[0][1] = glm::sin(rotation.z);
	rz[1][1] = glm::cos(rotation.z);

	rotationMat = rx*ry*rz;

	glm::mat4 scaleMat = glm::mat4();
	scaleMat[0][0] = scale.x;
	scaleMat[1][1] = scale.y;
	scaleMat[2][2] = scale.z;

	return translationMat * rotationMat * scaleMat;
}

__host__ __device__ Vector3f generateDisplacedOrigin(const Vector3f& normal, Vector3f& intersectionPoint,
													 Vector2f& sample, float& sampledDistance)
{
	//sample the squareToConcentricDiscExponentialFallOffDistribution function to generate a sample in a radial disc 
	//with the ditribution of that exponential function and then transform that to world space.
	float expDistRadialCoeff = 1 / (1 + sampledDistance);
	Point3f discPointExpDistribution = squareToConcentricDiscExponentialFallOffDistribution(sample, expDistRadialCoeff);

	Vector3f local_up = Vector3f(0, 1, 0);
	Vector3f world_normal = glm::normalize(-normal);

	//https://stackoverflow.com/questions/15101103/euler-angles-between-two-3d-vectors
	//http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/index.htm
	//https://stackoverflow.com/questions/16970863/is-yaw-pitch-and-roll-respectively-the-same-thing-as-heading-pitch-and-bank
	Vector3f crossproduct = glm::cross(local_up, world_normal);
	float cosRotAngle = glm::dot(local_up, world_normal); //because both vetors have length == 1
	float rotAngle = glm::acos(cosRotAngle);
	//you now have axis angle form convert to euler angles
	Vector3f eulerangles = axisAngletoEuler(crossproduct, rotAngle);

	Vector3f rotation = eulerangles;
	Vector3f translation = intersectionPoint;
	Vector3f scale = Vector3f(1.0f, 1.0f, 1.0f);
	glm::mat4 transform = buildTransformationMatrix(translation, rotation, scale);

	Vector3f newRayOrigin = glm::vec3(transform*glm::vec4(discPointExpDistribution, 1.0f));

	//move newRayOrigin into the object
	newRayOrigin += -glm::normalize(normal)*sampledDistance;

	return Vector3f(0.0f); //newRayOrigin;
}

__host__ __device__ Color3f f_Subsurface(const Material &m, float& samplePoint)
{
	float sampledDistance = sampleScatterDistance(samplePoint, m.scatteringCoefficient);
	float expColorDecay = 1 / (1 + sampledDistance); // same as expDistRadialCoeff
	return m.color*expColorDecay;
}

__host__ __device__ float pdf_Subsurface(const Vector3f& wo, Vector3f& wi, const float& thetaMin)
{
	float normalizedthetaMin = thetaMin / 180.0f;
	float g = smootherstep(normalizedthetaMin);

	float pdf = HenyeyGreensteinPhaseFunction(wo, wi, g);
	return pdf;
}

__host__ __device__ bool sample_f_Subsurface(const Vector3f& wo, Vector3f& sample, 
											const Material& m, Geom& geom,
											PathSegment& pathsegment, Geom * geoms,
											const Vector3f& normal, Vector3f& intersectionPoint,											
											Color3f& sampledColor, Vector3f& wi, float& pdf)
{
	//determine the sampleDistance
	//generate a ray whose origin is a point on a disc(with exponential distribution) generated around the surfaceNormal
	//and shifted inside the object by glm::normalize(normal)*sampledDistance
	//and this ray has a direction determined by the generateScatteredDirectionandPDF
	//carry out an intersection test with this new ray that originates from within the object
	//if sampleDistance<t the material was too thick --> have the material sampler sample another bxdf
	//ie set the pdf back to zero and and color to zero and sample another bxdf
	//else shift the exit point to the surface of the object, ie make sampledDistance = t
	//set the color to a scaled down value colorised by the material and set the pdf and wi

	float scatteringCoefficient = m.scatteringCoefficient;
	float thetaMin = m.thetaMin;

	Ray ray;
	Vector2f sample2f = Vector2f(sample2f[0], sample2f[1]);
	float sampledDistance = sampleScatterDistance(sample[2], scatteringCoefficient);
	ray.origin = generateDisplacedOrigin(normal, intersectionPoint, sample2f, sampledDistance);
	generateScatteredDirectionandPDF(sample2f, thetaMin, pdf, wi, wo);
	ray.direction = wi;

	ShadeableIntersection isx;
	computeIntersectionsWithSelectedObject(ray, geom, isx);

	if (sampledDistance < isx.t)
	{
		//return black
		sampledColor = Color3f(0.0f);
		return false;
	}

	//shift exit point to the surface of the object and check wi against the normal at that point
	if (isx.t >= 0.0f)
	{
		//actually hit the object
		intersectionPoint = isx.intersectPoint + EPSILON*2.0f*normal;
	}
	else
	{
		//didnt hit the object but should still shift the original intersection point
		intersectionPoint = ray.origin;
	}

	if (glm::dot(wi, normal)<0.0f)
	{
		//wi and normal are not in the same hemisphere of directions so invert wi
		wi = -wi;
	}

	float expColorDecay = 1 / (1 + sampledDistance); // same as expDistRadialCoeff
	sampledColor = m.color* expColorDecay*sampledDistance;
	//wi -- already set in generateScatteredDirectionandPDF
	//pdf -- already set in generateScatteredDirectionandPDF

	return true;
}

__host__ __device__ bool sample_f_Subsurface_second(const Vector3f& wo, Vector3f& sample,
											const Material& m, Geom& geom,
											PathSegment& pathSegment, Geom * geoms, int numGeoms,
											const Vector3f& normal, ShadeableIntersection& intersection,
											Color3f& sampledColor, Vector3f& wi, float& pdf,
											thrust::default_random_engine &rng)
{
	Ray nextRay;
	nextRay.origin = intersection.intersectPoint + EPSILON * pathSegment.ray.direction;
	nextRay.direction = pathSegment.ray.direction;

	PathSegment offsetPath = pathSegment;
	offsetPath.ray = nextRay;
	offsetPath.color = glm::vec3(0.f);

	float density = m.density;

	ShadeableIntersection prevIsect = intersection;
	ShadeableIntersection final;

	int	 maxBounce = 5;
	while (maxBounce > 0) {
		// Get the end point of the path 
		// This should still be the same object
		ShadeableIntersection end;
		computeIntersectionsForASingleRay(offsetPath, geoms, numGeoms, end);

		// Should always be the case.
		if (end.t > 0.f) {
			glm::vec3 path = end.intersectPoint - nextRay.origin;

			thrust::uniform_real_distribution<float> u01(0, 1);
			thrust::uniform_real_distribution<float> u(-1, 1);

			// Sample the medium for distance
			float ln = logf(u01(rng));
			float distanceTraveled = -ln / density;

			// If the sampled distance is less than the ray then we want to 
			// get a new direction for the next ray and add to the color.
			if (distanceTraveled < path.length()) {

				nextRay.origin = nextRay.origin + glm::normalize(path) * distanceTraveled;
				// Sample the medium for a direction
				nextRay.direction = SphereSample(rng);
				offsetPath.ray = nextRay;

				float transmission = expf(-density * distanceTraveled);

				// Color gets more muted as we go further along the ray.
				offsetPath.color += m.color * transmission*50000.0f;

				prevIsect = end;
			}
			else {
				final = end;
				break;
			}
		}
		else {
			final = prevIsect;
			break;
		}

		maxBounce--;
	}

	pathSegment.ray = nextRay;
	pathSegment.color = offsetPath.color;

	return true;
}