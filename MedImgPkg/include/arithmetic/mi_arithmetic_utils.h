#ifndef MEDIMGARITHMETIC_MI_ARITHMETIC_UTILS_H
#define MEDIMGARITHMETIC_MI_ARITHMETIC_UTILS_H

#include "arithmetic/mi_ellipsoid.h"
#include "arithmetic/mi_matrix4.h"
#include "arithmetic/mi_matrix4f.h"
#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_sphere.h"
#include "arithmetic/mi_vector3.h"
#include "arithmetic/mi_vector3f.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export ArithmeticUtils {
public:
    static Matrix4f convert_matrix(const Matrix4& mat);
    static Vector3f convert_vector(const Vector3& v);
    static Vector3f convert_point(const Point3& v);

    static Point2 dc_to_ndc(const Point2& pt_dc, int width, int height);
    // spill_tag 1 X direction 2 Y direction 0 no spill
    static Point2 ndc_to_dc(Point2 pt_ndc, int width, int height, int spill_tag); 
    static Point2 ndc_to_dc(Point2 pt_ndc, int width, int height);
    static Point2 ndc_to_dc_decimal(Point2 pt_ndc, int width, int height);

    // Check pt in [ (0,0,0) , bound ]
    static bool check_in_bound(const Point3& pt, const Point3& bound); 

    //0 inside bounding
    //-1 outside bounding
    static int get_valid_region(const unsigned int (&dim)[3],
                                 const Sphere& sphere, unsigned int (&begin)[3],
                                 unsigned int (&end)[3]);
    static int get_valid_region(const unsigned int (&dim)[3],
                                 const Ellipsoid& ellipsoid,
                                 unsigned int (&begin)[3],
                                 unsigned int (&end)[3]);

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif