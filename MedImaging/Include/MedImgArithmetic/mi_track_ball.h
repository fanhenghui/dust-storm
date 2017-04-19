#ifndef ARITHMETIC_TRACK_BALL_H_
#define ARITHMETIC_TRACK_BALL_H_

#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_quat4.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export TrackBall
{
public:
    /// \ Mouse position coordinate (0 ~ m_iWidth-1 , 0 ~ m_iHeight - 1)
    static Quat4 MouseMotionToRotation(
        const Point2& ptMouseFrom, const Point2& ptMouseTo,
        double width, double height, const Point2& ptBottomLeftCorner = Point2::kZeroPoint);

protected:
    ///Convert 2D window coordinates to coordinates on the 3D unit sphere. 
    static Point3 ConvertScreenPointToSphere_i(
        const Point2& ptPos, double width , double height , const Point2& ptBottomLeftCorner );

    static Quat4 CalculateTrackBallRotation(const Point3& ptPrevious , const Point3& ptCurrent , const Point3& ptCenter);
private:
};

MED_IMAGING_END_NAMESPACE
#endif

