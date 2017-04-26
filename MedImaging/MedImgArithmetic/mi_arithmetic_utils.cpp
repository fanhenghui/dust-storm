#include "mi_arithmetic_utils.h"

MED_IMAGING_BEGIN_NAMESPACE


Matrix4f ArithmeticUtils::convert_matrix(const Matrix4& mat)
{
    Vector4f col0((float)mat.m[0][0] , (float)mat.m[0][1] , (float)mat.m[0][2] , (float)mat.m[0][3]);
    Vector4f col1((float)mat.m[1][0] , (float)mat.m[1][1] , (float)mat.m[1][2] , (float)mat.m[1][3]);
    Vector4f col2((float)mat.m[2][0] , (float)mat.m[2][1] , (float)mat.m[2][2] , (float)mat.m[2][3]);
    Vector4f col3((float)mat.m[3][0] , (float)mat.m[3][1] , (float)mat.m[3][2] , (float)mat.m[3][3]);

    return Matrix4f(col0 , col1 , col2 , col3);
}

Vector3f ArithmeticUtils::convert_vector(const Vector3& v)
{
    return Vector3f((float)v.x , (float)v.y ,(float)v.z);
}

Vector3f ArithmeticUtils::convert_point(const Point3& v)
{
    return Vector3f((float)v.x , (float)v.y ,(float)v.z);
}

//double ArithmeticUtils::RoundDouble(double x)
//{
//    static const double magic = 6755399441055744.0; // (1<<51) | (1<<52)
//    double tmp = x;
//    tmp += magic;
//    return tmp;
//}
//
//int ArithmeticUtils::RoundInt( double x)
//{
//    static const double magic = 6755399441055744.0; // (1<<51) | (1<<52)
//    double tmp = x;
//    tmp += magic;
//    return *(int*)&tmp;
//}
//
//double ArithmeticUtils::FloorDouble(double x)
//{
//    static const double magic = 6755399441055744.0; // (1<<51) | (1<<52)
//    double tmp = x;
//    tmp += (x > 0) ? -0.499999999999 : +0.499999999999; //如果需要4舍5入取整就去掉这一行
//    tmp += magic;
//    return tmp;
//}
//
//int ArithmeticUtils::FloorInt(double x)
//{
//    static const double magic = 6755399441055744.0; // (1<<51) | (1<<52)
//    double tmp = x;
//    tmp += (x > 0) ? -0.499999999999 : +0.499999999999; //如果需要4舍5入取整就去掉这一行
//    tmp += magic;
//    return *(int*)&tmp;
//}

Point2 ArithmeticUtils::dc_to_ndc(const Point2& pt_dc , int width , int height)
{
    double x = (pt_dc.x +0.5)/(double)width;
    double y = (pt_dc.y +0.5)/(double)height;
    return Point2(x*2.0-1.0 , -(y*2.0 - 1.0));
}

Point2 ArithmeticUtils::ndc_to_dc(Point2 pt_ndc , int width , int height , int spill_tag)
{
    spill_tag = 0;
    if (pt_ndc.x < -1.0)
    {
        pt_ndc.x = -1.0;
        spill_tag = 1; 
    }
    else if (pt_ndc.x > 1.0)
    {
        pt_ndc.x = 1.0;
        spill_tag = 1; 
    }

    if (pt_ndc.y < -1.0)
    {
        pt_ndc.y = -1.0;
        spill_tag = 2; 
    }
    else if (pt_ndc.y > 1.0)
    {
        pt_ndc.y = 1.0;
        spill_tag = 2; 
    }

    double x = (pt_ndc.x+1.0)*0.5;
    double y = (-pt_ndc.y+1.0)*0.5;
    x = x*(double)width - 0.5;
    y = y*(double)height - 0.5;

    return Point2( (int)x , (int)y);
}

Point2 ArithmeticUtils::ndc_to_dc(Point2 pt_ndc , int width , int height)
{
    double x = (pt_ndc.x+1.0)*0.5;
    double y = (-pt_ndc.y+1.0)*0.5;
    x = x*(double)width - 0.5;
    y = y*(double)height - 0.5;

    return Point2( (int)x , (int)y);
}

Point2 ArithmeticUtils::ndc_to_dc_decimal(Point2 pt_ndc , int width , int height)
{
    double x = (pt_ndc.x+1.0)*0.5;
    double y = (-pt_ndc.y+1.0)*0.5;
    x = x*(double)width;
    y = y*(double)height;

    return Point2( x , y);
}

bool ArithmeticUtils::check_in_bound(const Point3& pt , const Point3& bound)
{
    if (pt.x < 0 || pt.x > bound.x || 
        pt.y < 0 || pt.y > bound.y || 
        pt.z < 0 || pt.z > bound.z)
    {
        return false;
    }
    else
    {
        return true;
    }
}

MED_IMAGING_END_NAMESPACE