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
    static Matrix4f ConvertMatrix(const Matrix4& mat);

    static Vector3f ConvertVector(const Vector3& v);

    static Vector3f ConvertPoint(const Point3& v);

    static Point2 DCToNDC(const Point2& ptDC ,  int iWidth , int iHeight);

    //iSpillTag 1 X direction 2 Y direction 0 no spill
    static Point2 NDCToDC(Point2 ptNDC , int iWidth , int iHeight , int iSpillTag);

    static Point2 NDCToDC(Point2 ptNDC , int iWidth , int iHeight );

    //TODO 下面的几个输出并不对
    static double FloorDouble(double x);

    static int FloorInt(double x);

    static double RoundDouble(double x);

    static int RoundInt(double x);

    //Check pt in [ (0,0,0) , vBound ]
    static bool CheckInBound(const Point3& pt , const Point3& vBound);

protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif