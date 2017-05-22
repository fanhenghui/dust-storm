#ifndef MED_IMAGING_INTERSECTION_TEST_H_
#define MED_IMAGING_INTERSECTION_TEST_H_

#include "Core/triangle.h"
#include "Core/rectangle.h"
#include "Core/line.h"
#include "Core/line_segment.h"
#include "Core/plane.h"
#include "Core/aabb.h"
#include "Core/sphere.h"

class  IntersectionTest
{
public:
    //平面相交，返回交线，平行或重合则检测失败
    static bool plane_to_plane(const Plane& plane1,const Plane& plane2,Line3D& intersected_line );

    ////平面相交三角形，返回交线
    //bool PlaneIntersectTriangle(const Plane& plane , const Triangle& triangle , LineSegment3D& line);

    ////平面相交三角形，返回两个三角形
    //bool PlaneIntersectTriangle(const Plane& plane , const Triangle& triangle , std::vector<Triangle> &vTriangle);

    ////平面相交四边形，返回交线
    //bool PlaneIntersectRectangle(const Plane& plane , const Rectangle& rect , LineSegment3D& line);

    ////平面相交AABB
    //bool PlaneIntersectAABB(const Plane& plane , const AABB& aabb);

    ////三重积
    //float ScalarTriple(const Vector3& u ,  const Vector3& v , const Vector3& w);

    ////两根线之间最近距离的线段
    //void NearestPointBetweenLine(const LineSegment3D& line1 , const LineSegment3D &line2 , bool &bParallel , Point3 &ptLine1, Point3 &ptLine2);

    //bool LineIntersectPlane(const LineSegment3D& line , const Plane& plane , Point3 &ptOut );

    ////bDiscard be true if line is in negative direction of the plane
    //bool LineIntersectPlane(const LineSegment3D& line , const Plane& plane , Point3 &ptOut , bool &bDiscard);

    //bool RayIntersectTriangle(const Point3 &pt0 , const Vector3& vDir , const Triangle& triangle , Point3 &ptOut);

    //bool RayIntersectRectangle(const Point3 &pt0 , const Vector3& vDir , const Rectangle& rect , Point3 &ptOut);


    //bool RayIntersectPlane(const Point3 &pt0 , const Vector3& vDir , const Plane& plane , Point3 &ptOut);

    //bool RayIntersectSphere(const Point3 &pt0 , const Vector3& vDir , const Sphere& sphere , Point3 &ptOutNear, Point3 &ptOutFar);

    //bool RayIntersectSphere( const Point3 &pt0 , const Vector3& vDir , const Sphere& sphere , Point3 &ptOutNear);

    ////bool RayIntersectAABB(const Point3 &pt0 , const Vector3& vDir , const Point);

    //bool Line2DIntersectLine2D(const Line2D line1,const Line2D line2,Point2& crossPoint);


protected:
private:
};



#endif