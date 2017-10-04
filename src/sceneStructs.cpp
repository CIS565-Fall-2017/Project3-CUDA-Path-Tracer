#include "sceneStructs.h"

// used for sorting matl is the single material that this intersection will use.
bool operator<(ShadeableIntersection &v1, ShadeableIntersection& v2)
{
	if (v1.matl != v2.matl) {
		return int(v1.matl) < int(v2.matl);
	}
	else {
		return v1.outside < v2.outside;
	}
}
