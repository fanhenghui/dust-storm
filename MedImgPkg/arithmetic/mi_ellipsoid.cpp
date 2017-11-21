#include "mi_ellipsoid.h"

MED_IMG_BEGIN_NAMESPACE

Ellipsoid::Ellipsoid() : _center(Point3::S_ZERO_POINT), _a(1), _b(1), _c(1)  {}

Ellipsoid::~Ellipsoid() {}

MED_IMG_END_NAMESPACE