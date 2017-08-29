#ifndef MEDIMGARITHMETIC_MI_INTERSECTION_TEST_H
#define MEDIMGARITHMETIC_MI_INTERSECTION_TEST_H

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
    static bool plane_to_plane(const Plane& plane1, const Plane& plane2,
                               Line3D& intersected_line);

    static bool aabb_to_aabb(const AABBI& l , const AABBI& r , AABBI& result);
    static bool aabb_to_aabb(const AABBUI& l , const AABBUI& r , AABBUI& result);

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif