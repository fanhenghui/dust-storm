#ifndef MED_IMAGING_ARITHMETIC_UTILS_H_
#define MED_IMAGING_ARITHMETIC_UTILS_H_


#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_matrix4f.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3.h"
#include "MedImgArithmetic/mi_vector3f.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export ArithmeticUtils
{
public:
    static Matrix4f convert_matrix(const Matrix4& mat);

    static Vector3f convert_vector(const Vector3& v);

    static Vector3f convert_point(const Point3& v);

    static Point2 dc_to_ndc(const Point2& pt_dc ,  int width , int height);

    //spill_tag 1 X direction 2 Y direction 0 no spill
    static Point2 ndc_to_dc(Point2 pt_ndc , int width , int height , int spill_tag);

    static Point2 ndc_to_dc(Point2 pt_ndc , int width , int height );

    //TODO 下面的几个输出并不对
    //static double FloorDouble(double x);

    //static int FloorInt(double x);

    //static double RoundDouble(double x);

    //static int RoundInt(double x);

    //Check pt in [ (0,0,0) , bound ]
    static bool check_in_bound(const Point3& pt , const Point3& bound);

protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif