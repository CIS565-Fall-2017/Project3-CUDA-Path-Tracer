#pragma once

#include "intersections.h"

// CHECKITOUT

////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Wraping Mode ///////////////////////////////
////////////////////////////////////////////////////////////////////////////

__host__ __device__ Point3f squareToDiskUniform(const Point2f &sample)
{
	float radius = glm::sqrt(sample.x);
	float phi = 2 * Pi * sample.y;
	return glm::vec3(radius * glm::cos(phi), radius * glm::sin(phi), 0.0f);
}

__host__ __device__ Point3f squareToDiskConcentric(const Point2f &sample)
{
	Point2f uOffset = 2.f * sample - Vector2f(1.0f, 1.0f);

	// Handle degeneracy at the origin
	if (uOffset.x == .0f && uOffset.y == .0f) return Point3f(0.0f, 0.0f, 0.0f);

	// Apply concentric mapping to point
	Float theta, r;
	
	if (glm::abs(uOffset.x) > glm::abs(uOffset.y))
	{
		r = uOffset.x;
		theta = Inv4Pi * (uOffset.y / uOffset.x);
	}
	else	
	{
		r = uOffset.y;
		theta = Inv2Pi - Inv4Pi * (uOffset.x / uOffset.y);
	}
	return r * Point3f(glm::cos(theta), glm::sin(theta), 0.0f);
}

__host__ __device__ float squareToDiskPDF(const Point3f &sample)
{
	return (float)InvPi;
}

__host__ __device__ Point3f squareToSphereUniform(const Point2f &sample)
{
	float z = 1 - 2 * sample.x;
	return glm::vec3(glm::sqrt(1 - z*z) *glm::cos(2 * Pi * sample.y), glm::sqrt(1 - z*z)*glm::sin(2 * Pi * sample.y), z);
}

__host__ __device__ float squareToSphereUniformPDF(const Point3f &sample)
{
	return (float)Inv4Pi;
}

__host__ __device__ Point3f squareToSphereCapUniform(const Point2f &sample, float thetaMin)
{
	thetaMin = glm::radians(180.f - thetaMin);
	float z = 1.f - (1.f - glm::cos(thetaMin)) * sample.x;
	return glm::vec3(glm::sqrt(1 - z*z) *glm::cos(2 * Pi * sample.y), glm::sqrt(1 - z*z)*glm::sin(2 * Pi * sample.y), z);
}

__host__ __device__ float squareToSphereCapUniformPDF(const Point3f &sample, float thetaMin)
{
	return (float)Inv2Pi  * (1.0f / (1.0f - glm::cos(glm::radians(180 - thetaMin))));
}

__host__ __device__ Point3f squareToHemisphereUniform(const Point2f &sample)
{
	float z = sample.x;
	float r = glm::sqrt(glm::max(0.0f, 1.0f - z * z));
	float phi = 2 * Pi * sample.y;
	return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), z);
}

__host__ __device__ float squareToHemisphereUniformPDF(const Point3f &sample)
{
	return (float)Inv2Pi;
}

__host__ __device__ Point3f squareToHemisphereCosine(const Point2f &sample)
{
	glm::vec3 disk = squareToDiskConcentric(sample);
	float x = disk.x;
	float y = disk.y;
	float z = glm::sqrt(glm::max(0.0f, 1.0f - x*x - y*y));

	return glm::vec3(x, y, z);
}

__host__ __device__ float squareToHemisphereCosinePDF(const Vector3f &normal, const Vector3f &wiW)
{
	float cosTheta = glm::dot(normal, wiW);
	return (float)InvPi * cosTheta;
}

__host__ __device__ void findNormalDirection(glm::vec3 normal, glm::vec3 &perpendicularDirection1, glm::vec3 &perpendicularDirection2)
{
	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (glm::abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (glm::abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));
}

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

	glm::vec3 perpendicularDirection1;
	glm::vec3 perpendicularDirection2;
	findNormalDirection(normal, perpendicularDirection1, perpendicularDirection2);

    return glm::normalize(up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2);
}


////////////////////////////////////////////////////////////////////////////
/////////////////////////////// MAth  Helper ///////////////////////////////
////////////////////////////////////////////////////////////////////////////


__host__ __device__ Vector3f Faceforward(const Vector3f &n, const Vector3f &v)
{
	return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__ bool IsBlack(const Color3f &Color)
{
	if (Color.r == 0.0f && Color.g == 0.0f && Color.b == 0.0f)
		return true;
	else
		return false;
}

__host__ __device__ bool SameHemisphere(const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW)
{
	return glm::dot(woW, normalW) * glm::dot(wiW, normalW) > 0.0f;
}



__host__ __device__ float CosTheta(const Vector3f &vector, const Vector3f &normalW)
{
	return glm::dot(vector, normalW);
}

__host__ __device__ Float Cos2Theta(const Vector3f &vector, const Vector3f &normalW)
{
	float cosTheta = CosTheta(vector, normalW);
	return cosTheta * cosTheta;
}

__host__ __device__ float AbsCosTheta(const Vector3f &vector, const Vector3f &normalW)
{
	return glm::abs(glm::dot(vector, normalW));
}

__host__ __device__ Float Sin2Theta(const Vector3f &w, const Vector3f &normalW)
{
	return glm::max(0.0f, 1.0f - Cos2Theta(w, normalW));
}

__host__ __device__ Float SinTheta(const Vector3f &w, const Vector3f &normalW)
{
	return glm::sqrt(Sin2Theta(w, normalW));
}

__host__ __device__ Float TanTheta(const Vector3f &w, const Vector3f &normalW)
{
	return SinTheta(w, normalW) / CosTheta(w, normalW);
}

__host__ __device__ Float Tan2Theta(const Vector3f &w, const Vector3f &normalW)
{
	return Sin2Theta(w, normalW) / Cos2Theta(w, normalW);
}


__host__ __device__ Float CosPhi(const Vector3f &w, const Vector3f &normalW)
{
	Float sinTheta = SinTheta(w, normalW);
	return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.0f, 1.0f);
}

__host__ __device__ Float SinPhi(const Vector3f &w, const Vector3f &normalW)
{
	Float sinTheta = SinTheta(w, normalW);
	return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.0f, 1.0f);
}

__host__ __device__ Float Cos2Phi(const Vector3f &w, const Vector3f &normalW)
{
	return CosPhi(w, normalW) * CosPhi(w, normalW);
}

__host__ __device__ Float Sin2Phi(const Vector3f &w, const Vector3f &normalW)
{
	return SinPhi(w, normalW) * SinPhi(w, normalW);
}

__host__ __device__ bool Refract(const Vector3f &wi, const Normal3f &n, Float eta,	Vector3f *wt)
{
	//using Snell's law
	Float cosThetaI = glm::dot(n, wi);
	Float sin2ThetaI = glm::max(Float(0.0f), Float(1.0f - cosThetaI * cosThetaI));
	Float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	if (sin2ThetaT >= 1.0f)
		return false;

	Float cosThetaT = glm::sqrt(1.0f - sin2ThetaT);
	*wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
	return true;
}

__host__ __device__ float PowerHeuristic(int nf, Float fPdf, int ng, Float gPdf)
{
	float Beta = 2.0f;
	float f = nf * fPdf;
	float g = ng * gPdf;
	float powf = glm::pow(f, Beta);
	float powg = glm::pow(g, Beta);

	if (powf + powg == 0.0f)
		return 0.0f;
	else
		return powf / (powf + powg);

}



__host__ __device__ Ray SpawnRay(const ShadeableIntersection &ref, const Vector3f &d)
{
	//float RayEpsilon = 0.000005f;

	Vector3f originOffset = ref.surfaceNormal * RayEpsilon;
	// Make sure to flip the direction of the offset so it's in
	// the same general direction as the ray direction
	originOffset = (glm::dot(d, ref.surfaceNormal) > 0) ? originOffset : -originOffset;
	Point3f o(ref.intersectPoint + originOffset);

	Ray newRay;
	newRay.origin = o;
	newRay.direction = d;

	return newRay;
}


////////////////////////////////////////////////////////////////
///////////////////////////// PLANE ////////////////////////////
////////////////////////////////////////////////////////////////

__host__ __device__ float getPlaneArea(const Geom &plane)
{
	return plane.scale.x * plane.scale.y;
}


__host__ __device__ ShadeableIntersection planeSample(const ShadeableIntersection &ref, const Point2f &xi, float *pdf, const Geom &shape)
{
	ShadeableIntersection inter;

	//A SquarePlane is assumed to have a radius of 0.5 and a center of <0,0,0>.
	Point2f pt = Point2f(xi.x - 0.5f, xi.y - 0.5f);

	glm::vec4 WSP = shape.transform * glm::vec4(pt.x, pt.y, 0.0f, 1.0f);

	Point3f WorldSpacePoint = Point3f(WSP.x, WSP.y, WSP.z);

	inter.intersectPoint = WorldSpacePoint;
	inter.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * glm::vec4(Normal3f(0.0f, 0.0f, 1.0f), 0.0f)));

	*pdf = 1.0f / getPlaneArea(shape);
	//*pdf = 1.0f;
	return inter;
}

////////////////////////////////////////////////////////////////
///////////////////////////// CUBE /////////////////////////////
////////////////////////////////////////////////////////////////

__host__ __device__ float getCubeArea(const Geom &cube, const glm::vec3 &surfaceNormal)
{
	glm::vec3 localNormal = multiplyMV(cube.inverseTransform, glm::vec4(surfaceNormal, 0.0f));

	if (glm::abs(localNormal.x) > 0.0f)
	{
		return cube.scale.z * cube.scale.y;
	}
	else if (glm::abs(localNormal.y) > 0.0f)
	{
		return cube.scale.x * cube.scale.z;
	}
	else
	{
		return cube.scale.x * cube.scale.y;
	}

	//return (cube.scale.x * cube.scale.y + cube.scale.y * cube.scale.z + cube.scale.x * cube.scale.z) * 2.0f;
	
}


__host__ __device__ ShadeableIntersection cubeSample(const ShadeableIntersection &ref, const Point2f &xi, float *pdf, const Geom &shape)
{
	
	Point2f pt = Point2f(xi.x - 0.5f, xi.y - 0.5f);
	glm::vec4 WSP;
	ShadeableIntersection inter;
	//Find nearest face

	glm::vec3 localPoint = multiplyMV(shape.inverseTransform, glm::vec4(ref.intersectPoint, 1.0f));	
	glm::vec3 gaplocalPoint = glm::abs(localPoint - glm::vec3(0.5f));

	if (gaplocalPoint.x < gaplocalPoint.z && gaplocalPoint.x < gaplocalPoint.y)
	{
		if (localPoint.x <= -0.5f)
		{
			WSP = shape.transform * glm::vec4(-0.5f, pt.x, pt.y, 1.0f);
			inter.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * glm::vec4(-1.0f, 0.0f, 0.0f, 0.0f)));
		}
		else if (localPoint.x >= 0.5f)
		{
			WSP = shape.transform * glm::vec4(0.5f, pt.x, pt.y, 1.0f);
			inter.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
		}
		//Inside
		else
		{
			*pdf = 0.0f;
			return inter;
		}
	}
	else if (gaplocalPoint.y < gaplocalPoint.x && gaplocalPoint.y < gaplocalPoint.z)
	{
		if (localPoint.y <= -0.5f)
		{
			WSP = shape.transform * glm::vec4(pt.x, -0.5f, pt.y, 1.0f);
			inter.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f)));
		}
		else if (localPoint.y >= 0.5f)
		{
			WSP = shape.transform * glm::vec4(pt.x, 0.5f, pt.y, 1.0f);
			inter.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
		}
		//Inside
		else
		{
			*pdf = 0.0f;
			return inter;
		}
	}
	else if (gaplocalPoint.z < gaplocalPoint.x && gaplocalPoint.z < gaplocalPoint.y)
	{
		if (localPoint.z <= -0.5f)
		{
			WSP = shape.transform * glm::vec4(pt.x, pt.y, -0.5f, 1.0f);
			inter.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
		}
		else if (localPoint.z >= 0.5f)
		{
			WSP = shape.transform * glm::vec4(pt.x, pt.y, 0.5f, 1.0f);
			inter.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));
		}
		//Inside
		else
		{
			*pdf = 0.0f;
			return inter;
		}
	}
	else
	{

	}

	
	Point3f WorldSpacePoint = Point3f(WSP.x, WSP.y, WSP.z);	
	inter.intersectPoint = WorldSpacePoint;

	

	*pdf = 1.0f / getCubeArea(shape, inter.surfaceNormal);
	//*pdf = 1.0f / getPlaneArea(shape);
	return inter;
}

////////////////////////////////////////////////////////////////
///////////////////////////// SPHERE ////////////////////////////
////////////////////////////////////////////////////////////////

__host__ __device__ float getSphereArea(const Geom &sphere)
{
	//DELETEME
	return 4.f * Pi * sphere.scale.x * sphere.scale.x; // We're assuming uniform scale
}

__host__ __device__ void CoordinateSystem(const Vector3f& v1, Vector3f* v2, Vector3f* v3)
{
	if (glm::abs(v1.x) > glm::abs(v1.y))
		*v2 = Vector3f(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
	else
		*v2 = Vector3f(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
	*v3 = glm::cross(v1, *v2);
}

__host__ __device__ ShadeableIntersection sphereSample(const Point2f &xi, Float *pdf, const Geom &shape)
{
	Point3f pObj = squareToSphereUniform(xi);
	float radius = 0.5f;

	pObj *= radius;

	ShadeableIntersection it;
	it.surfaceNormal = glm::normalize(multiplyMV(shape.inverseTransform, glm::vec4(pObj, 0.0f)));	          
	it.intersectPoint = Point3f(shape.transform * glm::vec4(pObj.x, pObj.y, pObj.z, 1.0f));

	*pdf = 1.0f / getSphereArea(shape);

	return it;
}

__host__ __device__ ShadeableIntersection sphereSample(const ShadeableIntersection &ref, const Point2f &xi, float *pdf, const Geom &shape)
{

	Point3f center = Point3f(shape.transform * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	Vector3f centerToRef = glm::normalize(center - ref.intersectPoint);
	Vector3f tan, bit;

	CoordinateSystem(centerToRef, &tan, &bit);

	float radius = 0.5f;

	Point3f pOrigin;
	if (glm::dot(center - ref.intersectPoint, ref.surfaceNormal) > 0)
		pOrigin = ref.intersectPoint + ref.surfaceNormal * RayEpsilon;
	else
		pOrigin = ref.intersectPoint - ref.surfaceNormal * RayEpsilon;

	if (glm::distance2(pOrigin, center) <= radius * radius) // Radius is 1, so r^2 is also 1
		return sphereSample(xi, pdf, shape);

	float sinThetaMax2 = radius / glm::distance2(ref.intersectPoint, center); // Again, radius is 1
	float cosThetaMax = std::sqrt(glm::max((float)0.0f, 1.0f - sinThetaMax2));
	float cosTheta = (1.0f - xi.x) + xi.x * cosThetaMax;
	float sinTheta = std::sqrt(glm::max((float)0.0f, 1.0f - cosTheta * cosTheta));
	float phi = xi.y * 2.0f * Pi;

	float dc = glm::distance(ref.intersectPoint, center);
	float ds = dc * cosTheta - glm::sqrt(glm::max((float)0.0f, 1 - dc * dc * sinTheta * sinTheta));

	float cosAlpha = (dc * dc + 1 - ds * ds) / (2 * dc * 1);
	float sinAlpha = glm::sqrt(glm::max((float)0.0f, 1.0f - cosAlpha * cosAlpha));

	Vector3f nObj = sinAlpha * glm::cos(phi) * -tan + sinAlpha * glm::sin(phi) * -bit + cosAlpha * -centerToRef;
	Point3f pObj = Point3f(nObj); // Would multiply by radius, but it is always 1 in object space

	ShadeableIntersection isect;

	pObj *= radius / glm::length(pObj); // pObj is already in object space with r = 1, so no need to perform this step

	isect.intersectPoint = Point3f(shape.transform * glm::vec4(pObj.x, pObj.y, pObj.z, 1.0f));
	isect.surfaceNormal = glm::normalize(multiplyMV(shape.inverseTransform, glm::vec4(nObj, 0.0f)));

	*pdf = 1.0f / (2.0f * Pi * (1 - cosThetaMax));

	return isect;
}


////////////////////////////////////////////////////////////////////////////
///////////////////////////// DiffuseAreaLight /////////////////////////////
////////////////////////////////////////////////////////////////////////////


__host__ __device__ Color3f Li(const glm::vec3 &isect_normal, const Vector3f &w, const Geom &shape, const Material *material)
{

	//if (twoSided)		
	Material Light = material[shape.materialid];

	return Light.color * Light.emittance;

}

__host__ __device__  ShadeableIntersection ShapeSample(const ShadeableIntersection &ref, const Point2f &xi, float *pdf, const Geom &shape)
{
	//TODO
	ShadeableIntersection inters;

	if (shape.type == CUBE)
	{
		inters = cubeSample(ref, xi, pdf, shape);
	}
	else if (shape.type == PLANE)
	{
		inters = planeSample(ref, xi, pdf, shape);
	}
	else if (shape.type == SPHERE)
	{
		inters = sphereSample(ref, xi, pdf, shape);
	}


	//Converet light sample weight to solid angle measure
	Vector3f LightVec = glm::normalize(inters.intersectPoint - ref.intersectPoint);

	float InvArea = *pdf;

	float NoL = glm::abs(glm::dot(inters.surfaceNormal, -LightVec));

	//Exception Handling
	if (NoL > 0.0f)
	{
		float dist = glm::distance(ref.intersectPoint, inters.intersectPoint);
		*pdf = dist * dist * InvArea / NoL;
	}
	else
		*pdf = 0.0f;

	return inters;
}

__host__ __device__ Color3f Sample_Li(const ShadeableIntersection &ref, const Point2f &xi, Vector3f *wiW, Float *pdf, const Geom &lightShape, const Material *material)
{
	ShadeableIntersection inter;

	inter = ShapeSample(ref, xi, pdf, lightShape);
	
	if(*pdf == 0.0f)
		return Color3f(0, 0, 0);

	*wiW = glm::normalize(inter.intersectPoint - ref.intersectPoint);

	if (glm::dot(*wiW, ref.surfaceNormal) < 0.0f)
		return Color3f(0, 0, 0);
		
	/*
	if (lightShape.type == CUBE)
	{
		float NoL = (glm::dot(inter.surfaceNormal, -*wiW));		

		if (NoL > 0.0f)
			*pdf = glm::distance2(ref.intersectPoint, inter.intersectPoint) / (NoL * getCubeArea(lightShape, inter.surfaceNormal));
		else
			*pdf = 0.0f;
	}
	else if (lightShape.type == PLANE)
	{
		float NoL = (glm::dot(inter.surfaceNormal, -*wiW));	

		if (NoL > 0.0f)
			*pdf = glm::distance2(ref.intersectPoint, inter.intersectPoint) / (NoL * getPlaneArea(lightShape));
		else
			*pdf = 0.0f;		
	}
	else if (lightShape.type == SPHERE)
	{
		float NoL = (glm::dot(inter.surfaceNormal, -*wiW));

		if (NoL > 0.0f)
			*pdf = glm::distance2(ref.intersectPoint, inter.intersectPoint) / (NoL * getSphereArea(lightShape));
		else
			*pdf = 0.0f;
	}
	else
		return Color3f(0, 0, 0);
	*/

	if (*pdf == 0.0f || (ref.intersectPoint == inter.intersectPoint))
		return Color3f(0, 0, 0);


	if (*pdf > 0)
		return Li(inter.surfaceNormal, -(*wiW), lightShape, material);
	else
		return Color3f(0, 0, 0);

}

__host__ __device__ float Pdf_Li(const ShadeableIntersection &ref, const Vector3f &wiW, const Geom &shape, int normalTexID, Image* imageHeader, glm::vec3 *imageData)
{

	Ray ray = SpawnRay(ref, wiW);
	//Intersection isectLight;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	glm::vec2 tmp_uv;
	bool outside = true;

	float t;

	if (shape.type == CUBE)
	{
		t = boxIntersectionTest(shape, ray, tmp_intersect, tmp_normal, tmp_uv, outside, normalTexID, imageHeader, imageData);
		if (t < 0.0f)
			return 0.0f;

		float NoL = (glm::dot(tmp_normal, -wiW));

		//if (this->twoSided)
		NoL = glm::abs(NoL);

		if (NoL > 0.0f)
			return glm::distance2(ref.intersectPoint, tmp_intersect) / (NoL * getCubeArea(shape, tmp_normal));
		else
			return 0.0f;

	}
	else if (shape.type == PLANE)
	{
		t = planeIntersectionTest(shape, ray, tmp_intersect, tmp_normal, tmp_uv, outside, normalTexID, imageHeader, imageData);
		if (t < 0.0f)
			return 0.0f;

		float NoL = (glm::dot(tmp_normal, -wiW));

		//if (this->twoSided)
		NoL = glm::abs(NoL);

		if (NoL > 0.0f)
			return glm::distance2(ref.intersectPoint, tmp_intersect) / (NoL * getPlaneArea(shape));
		else
			return 0.0f;
	}
	else if (shape.type == SPHERE)
	{
		t = sphereIntersectionTest(shape, ray, tmp_intersect, tmp_normal, tmp_uv, outside, normalTexID, imageHeader, imageData);
		if (t < 0.0f)
			return 0.0f;

		float NoL = (glm::dot(tmp_normal, -wiW));

		//if (this->twoSided)
		NoL = glm::abs(NoL);

		if (NoL > 0.0f)
			return glm::distance2(ref.intersectPoint, tmp_intersect) / (NoL * getSphereArea(shape));
		else
			return 0.0f;
	}
	else
		return 0.0f;
}

	


////////////////////////////////////////////////////////////////////////////
/////////////////////////////// IDLE DIFFUSE ///////////////////////////////
////////////////////////////////////////////////////////////////////////////

__host__ __device__ Color3f diffuse_f(const Material *material)
{
	return material->color * InvPi;
}

__host__ __device__ float diffuse_Pdf(const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW)
{
	if (SameHemisphere(woW, wiW, normalW))
		return squareToHemisphereCosinePDF(normalW, wiW);
	else
		return 0.0f;
}

__host__ __device__ Color3f diffuse_Sample_f(const Material *material, const Vector3f &woW, Vector3f *wiW, const Vector3f &normalW, thrust::default_random_engine &rng, float *pdf)
{
	//Generate Ingoing ray
	*wiW = glm::normalize(calculateRandomDirectionInHemisphere(normalW, rng));
	*pdf = 0.0f;


	if (SameHemisphere(woW, *wiW, normalW))
	{
		*pdf = diffuse_Pdf(woW, *wiW, normalW);
		return diffuse_f(material);
	}
	else
	{
		return glm::vec3(0.0f);
	}
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MIRROR //////////////////////////////////
////////////////////////////////////////////////////////////////////////////

__host__ __device__ Color3f mirror_f(const Material *material)
{
	return material->color;
}

__host__ __device__ float mirror_Pdf(const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW)
{
	if (SameHemisphere(woW, wiW, normalW))
		return 1.0f;
	else
		return 0.0f;
}

__host__ __device__ Color3f mirror_Sample_f(const Material *material, const Vector3f &woW, Vector3f *wiW, const Vector3f &normalW, thrust::default_random_engine &rng, float *pdf)
{
	*wiW = glm::reflect(-woW, normalW);

	if (SameHemisphere(woW, *wiW, normalW))
	{
		*pdf = mirror_Pdf(woW, *wiW, normalW);
		return mirror_f(material);
	}
	else
	{
		*pdf = 0.0f;
		return glm::vec3(0.0f);
	}
}




////////////////////////////////////////////////////////////////////////////
////////////////////////////// MicrofacetBRDF //////////////////////////////
////////////////////////////////////////////////////////////////////////////

__host__ __device__ Color3f SchlickFresnel(const Material *material, float cosTheta)
{
	return material->specular.color + glm::pow(1.0f - cosTheta, 5.0f) * (Color3f(1.0f) - material->specular.color);
}

__host__ __device__ float GGXDistribution_D(const Vector3f &whW, const Vector3f &normalW, float Roughness)
{
	float tan2Theta = Tan2Theta(whW, normalW);
	if (glm::isinf(tan2Theta))
		return 0.f;

	//float denom = (1.0f / (Cos2Phi(whW) / (alphax * alphax) + Sin2Phi(whW) / (alphay * alphay) + Cos2Theta(wh)));

	float denom = (1.0f / (Cos2Phi(whW, normalW) / (Roughness * Roughness) + Sin2Phi(whW, normalW) / (Roughness * Roughness) + Cos2Theta(whW, normalW)));
	
	//return (1.0f / (Pi* alphax * alphay)) * (denom*denom);
	return (1.0f / (Pi* Roughness * Roughness)) * (denom*denom);
}

__host__ __device__ float GGXDistribution_Lambda(const Vector3f &w, const Vector3f &normalW, float Roughness)
{
	float absTanTheta = glm::abs(TanTheta(w, normalW));
	if (glm::isinf(absTanTheta))
		return 0.;

	return glm::sqrt(1 + Roughness*Roughness*Tan2Theta(w, normalW));
}

__host__ __device__ float GGXDistribution_G(const Vector3f &woW, const Vector3f &normalW, float Roughness)
{
	return 2.0f / (1.0f + GGXDistribution_Lambda(woW, normalW, Roughness));
}

__host__ __device__ Vector3f GGXDistribution_Sample_wh(const Vector3f &woW, Vector3f *wiW, const Vector3f &normalW, thrust::default_random_engine &rng, float Roughness)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	Vector3f whW;
	float cosTheta = 0, phi = (2 * Pi) * u01(rng);

	//Only for isotrapic now
	//if (true alphax == alphay)
	{
		float xi0 = u01(rng);
		float tanTheta2 = Roughness * Roughness * xi0 / (1.0f - xi0);
		cosTheta = 1.0f / glm::sqrt(1.0f + tanTheta2);
	}
	/*
	else {
		phi =
			std::atan(alphay / alphax * std::tan(2 * Pi * xi[1] + .5f * Pi));
		if (xi[1] > .5f) phi += Pi;
		float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
		const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
		const float alpha2 =
			1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
		float tanTheta2 = alpha2 * xi[0] / (1 - xi[0]);
		cosTheta = 1 / std::sqrt(1 + tanTheta2);
	}
	*/
	float sinTheta = glm::sqrt(glm::max((float)0., (float)1. - cosTheta * cosTheta));

	glm::vec3 perpendicularDirection1;
	glm::vec3 perpendicularDirection2;
	findNormalDirection(normalW, perpendicularDirection1, perpendicularDirection2);

	whW = (sinTheta * glm::cos(phi))*perpendicularDirection1 + (sinTheta * glm::sin(phi)) *perpendicularDirection2 + cosTheta * normalW;
	whW = normalize(whW);

	//wh = Vector3f(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);

	if (!SameHemisphere(woW, whW, normalW))
		whW = -whW;

	return whW;
}

__host__ __device__ Color3f MicrofacetBRDF_f(const Material *material, const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW, float Roughness)
{
	float cosThetaO = AbsCosTheta(woW, normalW);
	float cosThetaI = AbsCosTheta(wiW, normalW);

	Vector3f whW = wiW + woW;

	if (cosThetaI == 0.0f || cosThetaO == 0.0f)
		return Color3f(0.f);

	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	if (whW.x == 0.0f || whW.y == 0.0f || whW.z == 0.0f)
		return Color3f(0.f);

	whW = glm::normalize(whW);

	Color3f Fresnel = SchlickFresnel(material, glm::dot(wiW, whW));

	/*
	if (MicrofacetBRDFModel == Torrance_Sparrow)
	{

		return R * distribution->D(wh) * distribution->G(wo, wi) * F / (4.0f * cosThetaI * cosThetaO);
	}
	//MicrofacetBRDFModel = COOK_TORRANCE
	else
	*/
	//MicrofacetBRDFModel = COOK_TORRANCE
	{
		/*
		float NdotH = CosTheta(whW, normalW);
		float NdotV = CosTheta(woW, normalW);
		float VdotH = glm::dot(woW, whW);
		float LdotH = glm::dot(wiW, whW);
		float NdotL = CosTheta(wiW, normalW);

		Fresnel = Fresnel + (Color3f(1.0f) - Fresnel)*glm::pow(1.0f - VdotH, 5.0f);

		float Gb = 2.0f * NdotH * NdotV / VdotH;
		float Gc = 2.0f * NdotH * NdotL / LdotH;
		float G = glm::min(1.0f, glm::min(Gb, Gc));
		*/

		return material->color * GGXDistribution_D(whW, normalW, Roughness) * GGXDistribution_G(woW, normalW, Roughness) * Fresnel / (Pi * cosThetaI * cosThetaO);
	}

}

__host__ __device__ float MicrofacetDistribution_Pdf(const Vector3f &whW, const Vector3f &normalW, float Roughness)
{
	return GGXDistribution_D(whW, normalW, Roughness) * AbsCosTheta(whW, normalW);
}


__host__ __device__ float MicrofacetBRDF_Pdf(const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW, float Roughness)
{   
	if (!SameHemisphere(woW, wiW, normalW))
		return 0.0f;

	Vector3f whW = glm::normalize(woW + wiW);
		
	float Dpdf = MicrofacetDistribution_Pdf(whW, normalW, Roughness);
	float Denom = (4.0f * glm::dot(woW, whW));
	float pdf = Dpdf / Denom;

	return pdf;
}

__host__ __device__ Color3f MicrofacetBRDF_Sample_f(const Material *material, const Vector3f &woW, Vector3f *wiW, const Vector3f &normalW, thrust::default_random_engine &rng,
	float *pdf, const float Roughness)
{
	
	Vector3f whW = GGXDistribution_Sample_wh(woW, wiW, normalW, rng, Roughness);

	*wiW = glm::normalize(glm::reflect(-woW, whW));

	if (!SameHemisphere(woW, *wiW, normalW))
	{
		return Color3f(1.0f, 1.0f, 0.0f);
	}

	*pdf = MicrofacetBRDF_Pdf(woW, *wiW, normalW, Roughness);

	return MicrofacetBRDF_f(material, woW, *wiW, normalW, Roughness);

}


////////////////////////////////////////////////////////////////////////////
/////////////////////////////// TRANSMISSION ///////////////////////////////
////////////////////////////////////////////////////////////////////////////

__host__ __device__ Color3f SpecularTransmission_Sample_f(const Material *material, const Vector3f &woW, Vector3f *wiW, const Vector3f &normalW, Float *pdf, float etaA, float etaB)
{
	bool entering = CosTheta(woW, normalW) > 0.0f;
	Float etaI = entering ? etaA : etaB;
	Float etaT = entering ? etaB : etaA;

	// Compute ray direction for specular transmission
	
	if (!Refract(woW, Faceforward(normalW, woW), etaI / etaT, wiW))
		return Color3f(0.0f);

	*pdf = 1.0f;
	//Color3f ft = material->specular.color *(Color3f(1.0f) - SchlickFresnel(material, CosTheta(*wiW, normalW)));
	Color3f ft = SchlickFresnel(material, CosTheta(*wiW, normalW));
	//return (Color3f(1.0f) - SchlickFresnel(material, CosTheta(*wiW, normalW)));
	return ft / AbsCosTheta(*wiW, normalW);
}

__host__ __device__ Color3f LambertianTransmission_f(const Material *material, const Vector3f &woW, const Vector3f &wiW) 
{
	return material->specular.color * InvPi;
}

__host__ __device__ Float LambertianTransmission_Pdf(const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW)
{
	return !SameHemisphere(woW, wiW, normalW) ? AbsCosTheta(wiW, normalW) * InvPi : 0.0f;
}

__host__ __device__ Color3f LambertianTransmission_Sample_f(const Material *material, const Vector3f &woW, Vector3f *wiW, const Vector3f &normalW, thrust::default_random_engine &rng, Float *pdf)
{
	

	*wiW = glm::normalize(calculateRandomDirectionInHemisphere(normalW, rng));

	//if (wo.z > 0) wi->z *= -1;
	
	
	if (glm::dot(woW, normalW) > 0.0f)
	{
		*wiW = glm::normalize(*wiW + -2.0f * glm::dot(normalW, *wiW) * normalW);		
	}
	

	*pdf = LambertianTransmission_Pdf(woW, *wiW, normalW);
	

	return LambertianTransmission_f(material, woW, *wiW);
}





////////////////////////////////////////////////////////////////////////////
////////////////////////////// MicrofacetBTDF //////////////////////////////
////////////////////////////////////////////////////////////////////////////

__host__ __device__ Color3f MicrofacetBTDF_f(const Material *material, const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW, const float Roughness, float eta)
{

	if (SameHemisphere(woW, wiW, normalW))
		return Color3f(0.0f);

	float cosThetaO = CosTheta(woW, normalW);
	float cosThetaI = CosTheta(wiW, normalW);
	if (cosThetaI == 0 || cosThetaO == 0)
		return Color3f(0.0f);


	//float eta = CosTheta(woW, normalW) > 0.0f ? (etaB / etaA) : (etaA / etaB);
	Vector3f whW = glm::normalize(woW + wiW * eta);
		
	if (glm::dot(whW, normalW) < 0.0f)
		whW = -whW;

	float WoDotWh = glm::dot(woW, whW);
	float WiDotWh = glm::dot(wiW, whW);

	Color3f Fresnel = SchlickFresnel(material, WoDotWh);

	float sqrtDenom = WoDotWh + eta * WiDotWh;

	return Fresnel * glm::abs(GGXDistribution_D(whW, normalW, Roughness) * GGXDistribution_G(woW, normalW, Roughness) * eta * eta * (glm::abs(WiDotWh) *glm::abs(WoDotWh)) / (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}

__host__ __device__ float MicrofacetBTDF_Pdf(const Vector3f &woW, const Vector3f &wiW, const Vector3f &normalW, const float Roughness, float eta)
{
	if (SameHemisphere(woW, wiW, normalW))
		return 0.0f;

	//float eta = CosTheta(woW, normalW) > 0 ? (etaB / etaA) : (etaA / etaB);
	Vector3f whW = glm::normalize(woW + wiW*eta);

	float sqrtDenom = glm::dot(woW, whW) + eta * glm::dot(wiW, whW);

	if (sqrtDenom <= 0.0f)
		return 0.0f;

	float dwh_dwi = glm::abs((eta * eta * glm::dot(wiW, whW)) / (sqrtDenom * sqrtDenom));

	return MicrofacetDistribution_Pdf(whW, normalW, Roughness) * dwh_dwi;
}

__host__ __device__ Color3f MicrofacetBTDF_Sample_f(const Material *material, const Vector3f &woW, Vector3f *wiW, const Vector3f &normalW, thrust::default_random_engine &rng,
	float *pdf, const float Roughness, float etaA, float etaB)
{
	float cosTheta = CosTheta(woW, normalW);

	if (cosTheta == 0.0f)
	{
		*pdf = 0.0f;
		return Color3f(0.0f);
	}

	/*
	bool bEntering = cosTheta > 0.0f;
	float etaI = bEntering ? etaA : etaB;
	float etaT = bEntering ? etaB : etaA;
	float eta = etaI / etaT;
	*/

	Float eta = CosTheta(woW, normalW) > 0.0f ? (etaA / etaB) : (etaB / etaA);

	Vector3f whW = GGXDistribution_Sample_wh(woW, wiW, normalW, rng, Roughness);

	if (!Refract(woW, whW, eta, wiW))
	{
		*pdf = 0.0f;
		return Color3f(0.0f);
	}
	
	*pdf = MicrofacetBTDF_Pdf(woW, *wiW, normalW, Roughness, 1.0f / eta);
	return MicrofacetBTDF_f(material, woW, *wiW, normalW, Roughness, 1.0f / eta);

}





/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
	    glm::vec3 woW,
	    glm::vec3 & wiW,
        glm::vec3 normal,
	    glm::vec3 & color,
	    float & pdf,
        const Material &m,
        thrust::default_random_engine &rng
	) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	float rayEpsilon = m.hasRefractive > 0.0f ? 0.001f : RayEpsilon;

	//reflection
	if (m.hasRefractive > 0.0f)
	{
		//Mirror reflection
		if (m.hasReflective == 1.0f)
			color = SpecularTransmission_Sample_f(&m, woW, &wiW, normal, &pdf, 1.0f, m.indexOfRefraction);
		//Lambert reflection
		else if(m.hasReflective == 0.0f)
			color = LambertianTransmission_Sample_f(&m, woW, &wiW, normal, rng, &pdf);
		//Microfacet reflection
		else
			color = MicrofacetBTDF_Sample_f(&m, woW, &wiW, normal, rng, &pdf, m.Roughness, 1.0f, m.indexOfRefraction);
		
	}
	//Idle diffuse
	else if (m.hasReflective == 0.0f)
	{
		color = diffuse_Sample_f(&m, woW, &wiW, normal, rng, &pdf);
	}
	// Mirror
	else if (m.hasReflective == 1.0f)
	{
		color = mirror_Sample_f(&m, woW, &wiW, normal, rng, &pdf);
	}
	//Microfacet PBRT
	else
	{
		color = MicrofacetBRDF_Sample_f(&m, woW, &wiW, normal, rng, &pdf, m.Roughness);
	}

	//new Ray
	pathSegment.ray.direction = wiW;
	pathSegment.ray.origin = intersect;

	Vector3f originOffset = normal * rayEpsilon;

	if (glm::dot(pathSegment.ray.direction, normal) < 0.0f)
		originOffset = -originOffset;

	pathSegment.ray.origin += originOffset;
}


__host__ __device__
void GetFandPDF(
	glm::vec3 intersect,
	glm::vec3 woW,
	glm::vec3 wiW,
	glm::vec3 normal,
	glm::vec3 & color,
	float & pdf,
	const Material &m	
) {	
	//Refractive
	if (m.hasRefractive > 0.0f)
	{
		//Mirror reflection
		if (m.hasReflective == 1.0f)
			color = SpecularTransmission_Sample_f(&m, woW, &wiW, normal, &pdf, 1.0f, m.indexOfRefraction);
		//Lambert reflection
		else if (m.hasReflective == 0.0f)
		{
			color = LambertianTransmission_f(&m, woW, wiW);
			pdf = LambertianTransmission_Pdf(woW, wiW, normal);
		}
		//Microfacet reflection
		else
		{
			float etaA = 1.0f;
			float etaB = m.indexOfRefraction;
			Float eta = CosTheta(woW, normal) > 0.0f ? (etaA / etaB) : (etaB / etaA);
			color = MicrofacetBTDF_f(&m, woW, wiW, normal, m.Roughness, eta);
			pdf = MicrofacetBTDF_Pdf(woW, wiW, normal, m.Roughness, eta);
		}

	}
	//Idle diffuse
	else if (m.hasReflective == 0.0f)
	{
		color = diffuse_f(&m);
		pdf = diffuse_Pdf(woW, wiW, normal);
	}
	// Mirror
	else if (m.hasReflective == 1.0f)
	{
		color = mirror_f(&m);
		pdf = mirror_Pdf(woW, wiW, normal);
	}
	//Microfacet PBRT
	else
	{
		color = MicrofacetBRDF_f(&m, woW, wiW, normal, m.Roughness);
		pdf = MicrofacetBRDF_Pdf(woW, wiW, normal, m.Roughness);
	}
}