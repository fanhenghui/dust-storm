#ifndef ARITHMETIC_ORTHO_CAMERA_H_
#define ARITHMETIC_ORTHO_CAMERA_H_

#include "MedImgArithmetic/mi_camera_base.h"
#include "MedImgArithmetic/mi_quat4.h"
#include "MedImgArithmetic/mi_vector2.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export OrthoCamera : public CameraBase
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

protected:
    virtual void calculate_projection_matrix_i();

private:
    double _Left;
    double _Right;
    double _Bottom;
    double _Top;
    double _Near;
    double _Far;
    Matrix4 _matProjection;
    bool _bIsPCalculated;

    //Zoom
    double _ZoomFactor;
    Vector2 _VecPan;
};

MED_IMAGING_END_NAMESPACE
#endif


