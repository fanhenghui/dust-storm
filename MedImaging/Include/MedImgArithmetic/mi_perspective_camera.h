#ifndef ARITHMETIC_PERSPECTIVE_CAMERA_H_
#define ARITHMETIC_PERSPECTIVE_CAMERA_H_

#include "MedImgArithmetic/mi_camera_base.h"
#include "MedImgArithmetic/mi_vector2.h"
#include "MedImgArithmetic/mi_matrix4.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export PerspectiveCamera : public CameraBase
{
public:
    PerspectiveCamera();

    PerspectiveCamera(double fovy, double aspect, double zNear, double zFar);

    virtual ~PerspectiveCamera();

    void set_perspective(double fovy, double aspect, double zNear, double zFar);

    void set_near_clip_distance(double zNear);

    void set_far_clip_distance(double zFar);

    virtual double get_near_clip_distance() const;

    virtual double get_far_clip_distance() const;

    void set_fovy(double fovy);

    void set_aspect_ratio(double aspect);

    virtual Matrix4 get_projection_matrix();

    virtual Matrix4 get_view_projection_matrix();

    virtual void zoom(double rate);

    virtual void pan(const Vector2& pan);

    PerspectiveCamera& operator =(const PerspectiveCamera& camera);

protected:
    virtual void calculate_projection_matrix_i();

private:
    double m_Fovy;
    double m_Aspect;
    double m_Near;
    double m_Far;
    Matrix4 m_matProjection;
    bool m_bIsPCalculated;
private:
};

MED_IMAGING_END_NAMESPACE
#endif