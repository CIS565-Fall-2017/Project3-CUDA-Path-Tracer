#include "FresnelDielectric.h"

float FrDielectric(float cosThetaI, float etaI, float etaT)
{
	cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

	//potentially swap indicieds of refraction
	bool entering = cosThetaI >0.f;

	if (!entering)
	{
		std::swap(etaI, etaT);
		cosThetaI = std::abs(cosThetaI);
	}

	//copmute cosThetaT using Snell's law
	float sinThetaI = std::sqrt(std::max((float)0, 1 - cosThetaI*cosThetaI));
	float sinThetaT = etaI / etaT*sinThetaI;

	//handle total internal reflection
	if (sinThetaT >= 1)
	{
		return 1.0f;
	}

	float cosThetaT = std::sqrt(std::max((float)0, 1 - sinThetaT*sinThetaT));

	float Rparl = ((etaT*cosThetaI) - (etaI*cosThetaT)) / ((etaT*cosThetaI) + (etaI*cosThetaT));
	float Rperp = ((etaI*cosThetaI) - (etaT*cosThetaT)) / ((etaI*cosThetaI) + (etaT*cosThetaT));

	return (Rparl*Rparl + Rperp*Rperp) / 2;
}

glm::vec3 FresnelDielectric::Evaluate(float cosThetaI) const
{
	return FrDielectric(cosThetaI, etaI, etaT)*glm::vec3(1.0f);
}
