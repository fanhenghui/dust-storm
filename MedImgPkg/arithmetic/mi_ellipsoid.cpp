#include "mi_ellipsoid.h"

MED_IMG_BEGIN_NAMESPACE

Ellipsoid::Ellipsoid() : _a(1), _b(1), _c(1), _center(Point3::S_ZERO_POINT) {}

Ellipsoid::~Ellipsoid() {}

MED_IMG_END_NAMESPACE