#include "mi_arithmetic_utils.h"

MED_IMAGING_BEGIN_NAMESPACE


Matrix4f ArithmeticUtils::convert_matrix(const Matrix4& mat)
{
    Vector4f vCol0((float)mat.m[0][0] , (float)mat.m[0][1] , (float)mat.m[0][2] , (float)mat.m[0][3]);
    Vector4f vCol1((float)mat.m[1][0] , (float)mat.m[1][1] , (float)mat.m[1][2] , (float)mat.m[1][3]);
    Vector4f vCol2((float)mat.m[2][0] , (float)mat.m[2][1] , (float)mat.m[2][2] , (float)mat.m[2][3]);
    Vector4f vCol3((float)mat.m[3][0] , (float)mat.m[3][1] , (float)mat.m[3][2] , (float)mat.m[3][3]);

    return Matrix4f(vCol0 , vCol1 , vCol2 , vCol3);
}

Vector3f ArithmeticUtils::convert_vector(const Vector3& v)
{
    return Vector3f((float)v.x , (float)v.y ,(float)v.z);
}

Vector3f ArithmeticUtils::ConvertPoint(const Point3& v)
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

Point2 ArithmeticUtils::dc_to_ndc(const Point2& ptDC , int iWidth , int iHeight)
{
    double x = (ptDC.x +0.5)/(double)iWidth;
    double y = (ptDC.y +0.5)/(double)iHeight;
    return Point2(x*2.0-1.0 , -(y*2.0 - 1.0));
}

Point2 ArithmeticUtils::ndc_to_dc(Point2 ptNDC , int iWidth , int iHeight , int iSpillTag)
{
    iSpillTag = 0;
    if (ptNDC.x < -1.0)
    {
        ptNDC.x = -1.0;
        iSpillTag = 1; 
    }
    else if (ptNDC.x > 1.0)
    {
        ptNDC.x = 1.0;
        iSpillTag = 1; 
    }

    if (ptNDC.y < -1.0)
    {
        ptNDC.y = -1.0;
        iSpillTag = 2; 
    }
    else if (ptNDC.y > 1.0)
    {
        ptNDC.y = 1.0;
        iSpillTag = 2; 
    }

    double x = (ptNDC.x+1.0)*0.5;
    double y = (-ptNDC.y+1.0)*0.5;
    x = x*(double)iWidth - 0.5;
    y = y*(double)iHeight - 0.5;

    return Point2( (int)x , (int)y);
}

Point2 ArithmeticUtils::ndc_to_dc(Point2 ptNDC , int iWidth , int iHeight)
{
    double x = (ptNDC.x+1.0)*0.5;
    double y = (-ptNDC.y+1.0)*0.5;
    x = x*(double)iWidth - 0.5;
    y = y*(double)iHeight - 0.5;

    return Point2( (int)x , (int)y);
}

bool ArithmeticUtils::check_in_bound(const Point3& pt , const Point3& vBound)
{
    if (pt.x < 0 || pt.x > vBound.x || 
        pt.y < 0 || pt.y > vBound.y || 
        pt.z < 0 || pt.z > vBound.z)
    {
        return false;
    }
    else
    {
        return true;
    }
}

MED_IMAGING_END_NAMESPACE