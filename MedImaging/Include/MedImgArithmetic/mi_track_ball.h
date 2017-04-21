#ifndef ARITHMETIC_TRACK_BALL_H_
#define ARITHMETIC_TRACK_BALL_H_

#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_quat4.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export TrackBall
{
public:
    /// \ Mouse position coordinate (0 ~ _width-1 , 0 ~ _height - 1)
    static Quat4 mouse_motion_to_rotation(
        const Point2& ptMouseFrom, const Point2& ptMouseTo,
        double width, double height, const Point2& ptBottomLeftCorner = Point2::S_ZERO_POINT);

protected:
    ///Convert 2D window coordinates to coordinates on the 3D unit sphere. 
    static Point3 convert_screen_point_to_sphere_i(
        const Point2& ptPos, double width , double height , const Point2& ptBottomLeftCorner );

    static Quat4 calculate_track_ball_rotation(const Point3& ptPrevious , const Point3& ptCurrent , const Point3& ptCenter);
private:
};

MED_IMAGING_END_NAMESPACE
#endif

