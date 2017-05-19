#ifndef ARITHMETIC_CAMERA_BASE_H_
#define ARITHMETIC_CAMERA_BASE_H_

#include "Core/vector2.h"
#include "Core/matrix4.h"
#include "Core/quat4.h"

class CameraBase
{
public:
    CameraBase();

    virtual ~CameraBase();

    void set_eye(const Point3& ptEye);

    void set_look_at(const Point3& ptCenter);

    void set_up_direction(const Vector3& vecUp);

    Point3 get_eye() const;

    Point3 get_look_at() const;

    Vector3 get_up_direction() const;

    Vector3 get_view_direction() const;

    Matrix4 get_view_matrix();

    virtual Matrix4 get_projection_matrix() = 0;

    virtual Matrix4 get_view_projection_matrix() = 0;

    virtual double get_near_clip_distance() const = 0;

    virtual double get_far_clip_distance() const = 0;

    void rotate(const Quat4& quat);

    void rotate(const Matrix4& mat);

    virtual void zoom(double rate) = 0;

    virtual void pan(const Vector2& pan) = 0;

    CameraBase& operator =(const CameraBase& camera);

    bool operator == (const CameraBase& camera) const; 

protected:
    void calculate_view_matrix_i();

    virtual void calculate_projection_matrix_i() = 0;

protected:
    Point3 _eye;
    Point3 _at;
    Vector3 _up;
    Matrix4 _mat_view;
    bool _is_view_mat_cal;

};



#endif
