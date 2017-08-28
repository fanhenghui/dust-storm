#ifndef MED_IMG_INTERSECTION_TEST_H_
#define MED_IMG_INTERSECTION_TEST_H_

#include "arithmetic/mi_triangle.h"
#include "arithmetic/mi_rectangle.h"
#include "arithmetic/mi_line.h"
#include "arithmetic/mi_line_segment.h"
#include "arithmetic/mi_plane.h"
#include "arithmetic/mi_aabb.h"
#include "arithmetic/mi_sphere.h"

MED_IMG_BEGIN_NAMESPACE 

class Arithmetic_Export IntersectionTest
{
public:
    //ƽ���ཻ�����ؽ��ߣ�ƽ�л��غ�����ʧ��
    static bool plane_to_plane(const Plane& plane1,const Plane& plane2,Line3D& intersected_line );

    ////ƽ���ཻ�����Σ����ؽ���
    //bool PlaneIntersectTriangle(const Plane& plane , const Triangle& triangle , LineSegment3D& line);

    ////ƽ���ཻ�����Σ���������������
    //bool PlaneIntersectTriangle(const Plane& plane , const Triangle& triangle , std::vector<Triangle> &vTriangle);

    ////ƽ���ཻ�ı��Σ����ؽ���
    //bool PlaneIntersectRectangle(const Plane& plane , const Rectangle& rect , LineSegment3D& line);

    ////ƽ���ཻAABB
    //bool PlaneIntersectAABB(const Plane& plane , const AABB& aabb);

    ////���ػ�
    //float ScalarTriple(const Vector3& u ,  const Vector3& v , const Vector3& w);

    ////������֮�����������߶�
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

MED_IMG_END_NAMESPACE

#endif