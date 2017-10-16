#include "mi_line.h"

MED_IMG_BEGIN_NAMESPACE

Line2D::Line2D() {}

Line2D::~Line2D() {}

void Line2D::to_func(double& a, double& b, double& c) const{
    a = _dir.x;
    b = _dir.y;
    c = a*_pt.x + b*_pt.y;
}

Line3D::Line3D() {}

Line3D::~Line3D() {}

MED_IMG_END_NAMESPACE