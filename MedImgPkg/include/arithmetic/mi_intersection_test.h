#ifndef MEDIMGARITHMETIC_INTERSECTION_TEST_H_
#define MEDIMGARITHMETIC_INTERSECTION_TEST_H_

#include "arithmetic/mi_aabb.h"
#include "arithmetic/mi_line.h"
#include "arithmetic/mi_line_segment.h"
#include "arithmetic/mi_plane.h"
#include "arithmetic/mi_rectangle.h"
#include "arithmetic/mi_sphere.h"
#include "arithmetic/mi_triangle.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export IntersectionTest {
public:
  static bool plane_to_plane(const Plane &plane1, const Plane &plane2,
                             Line3D &intersected_line);

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif