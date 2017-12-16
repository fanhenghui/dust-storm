#ifndef MEDIMGARITHMETIC_MI_PERSPECTIVE_CAMERA_H
#define MEDIMGARITHMETIC_MI_PERSPECTIVE_CAMERA_H

#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_matrix4.h"
#include "arithmetic/mi_vector2.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export PerspectiveCamera : public CameraBase {
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

    PerspectiveCamera& operator=(const PerspectiveCamera& camera);

protected:
    virtual void calculate_projection_matrix();

private:
    double _Fovy;
    double _Aspect;
    double _Near;
    double _Far;
    Matrix4 _matProjection;
    bool _bIsPCalculated;

private:
};

MED_IMG_END_NAMESPACE
#endif