#ifndef ARITHMETIC_ORTHO_CAMERA_H_
#define ARITHMETIC_ORTHO_CAMERA_H_

#include "Core/camera_base.h"
#include "Core/quat4.h"
#include "Core/vector2.h"

class  OrthoCamera : public CameraBase
{
public:
    OrthoCamera();

    OrthoCamera(double left, double right, double bottom, double top, double near, double far0);

    virtual ~OrthoCamera();

    void set_ortho(double left, double right, double bottom, double top, double near, double far0);

    void get_ortho(double& left, double& right, double& bottom, double& top, double& near, double& far0) const;

    virtual Matrix4 get_projection_matrix();

    virtual Matrix4 get_view_projection_matrix();

    virtual void zoom(double rate);

    virtual void pan(const Vector2& pan);

    double get_near_clip_distance() const;

    double get_far_clip_distance() const;

    OrthoCamera& operator =(const OrthoCamera& camera);

    bool operator == (const OrthoCamera& camera) const;

protected:
    virtual void calculate_projection_matrix_i();

private:
    double _left;
    double _right;
    double _bottom;
    double _top;
    double _near;
    double _far;
    Matrix4 _mat_projection;
    bool _is_proj_mat_cal;

    //Zoom
    double _ZoomFactor;
    Vector2 _VecPan;
};


#endif


