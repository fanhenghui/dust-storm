#ifndef MEDIMGARITHMETIC_MI_ORTHO_CAMERA_H
#define MEDIMGARITHMETIC_MI_ORTHO_CAMERA_H

#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_quat4.h"
#include "arithmetic/mi_vector2.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export OrthoCamera : public CameraBase {
public:
    OrthoCamera();

    OrthoCamera(double left, double right, double bottom, double top, double near,
                double far0);

    virtual ~OrthoCamera();

    void set_ortho(double left, double right, double bottom, double top,
                   double near, double far0);

    void get_ortho(double& left, double& right, double& bottom, double& top,
                   double& near, double& far0) const;

    virtual Matrix4 get_projection_matrix();

    virtual Matrix4 get_view_projection_matrix();

    virtual void zoom(double rate);

    double get_near_clip_distance() const;

    double get_far_clip_distance() const;

    OrthoCamera& operator=(const OrthoCamera& camera);

    bool operator==(const OrthoCamera& camera) const;

    bool operator!=(const OrthoCamera& camera) const;

protected:
    virtual void calculate_projection_matrix();

private:
    double _left;
    double _right;
    double _bottom;
    double _top;
    double _near;
    double _far;
    Matrix4 _mat_projection;
    bool _is_proj_mat_cal;

    // Zoom
    double _zoom_factor;
};

MED_IMG_END_NAMESPACE
#endif
