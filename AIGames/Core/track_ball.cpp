#include "track_ball.h"
#include "vector2.h"
#include "vector3.h"




Point3 TrackBall::convert_screen_point_to_sphere_i( const Point2& ptPos, double width , double height , const Point2& ptBottomLeftCorner )
{
    double x = (ptPos.x - ptBottomLeftCorner.x) / (width/2)-1;
    double y = 1-(ptPos.y - ptBottomLeftCorner.y) / (height/2);

    double z2 = 1.0 - x*x - y*y;
    double z = z2 > 0.5 ? sqrt(z2) : 0.5/sqrt(x*x + y*y);

    return Point3(x,y,z);
}

Quat4 TrackBall::mouse_motion_to_rotation( const Point2& ptMouseFrom, const Point2& ptMouseTo, double width, double height, const Point2& ptBottomLeftCorner )
{
    if ((ptMouseFrom.x > ptBottomLeftCorner.x) &&
        (ptMouseFrom.x < ptBottomLeftCorner.x + width) &&
        (ptMouseFrom.y > ptBottomLeftCorner.y) &&
        (ptMouseFrom.y < ptBottomLeftCorner.y + height) )
    {
        const Point3 ptMouseFrom3D = convert_screen_point_to_sphere_i(ptMouseFrom , width , height , ptBottomLeftCorner);
        const Point3 ptMouseTo3D = convert_screen_point_to_sphere_i(ptMouseTo , width , height , ptBottomLeftCorner);

        Quat4 qRotation = calculate_track_ball_rotation(ptMouseFrom3D , ptMouseTo3D , Point3::S_ZERO_POINT);
        return qRotation;
    }
    else
    {
        return Quat4(0.0 , Vector3(1,0,0));
    }
}

Quat4 TrackBall::calculate_track_ball_rotation( const Point3& ptPrevious , const Point3& ptCurrent , const Point3& ptCenter )
{
    const Vector3 v0 = ptCurrent - ptCenter;
    const Vector3 v1 = ptPrevious - ptCenter;
    const Vector3 vAxis = cross(v0 , v1);
    const double dAngle = angle_between(v0 , v1);
    return Quat4(dAngle , vAxis);
}


