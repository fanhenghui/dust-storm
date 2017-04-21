#include "mi_ortho_camera.h"

MED_IMAGING_BEGIN_NAMESPACE

    OrthoCamera::OrthoCamera() : CameraBase()
    , _bIsPCalculated(false)
    , _Left(0), _Right(0), _Bottom(0), _Top(0), _Near(0), _Far(0), _ZoomFactor(1.0)
{
    _matProjection = Matrix4::S_IDENTITY_MATRIX;
    _VecPan = Vector2::S_ZERO_VECTOR;
}

OrthoCamera::OrthoCamera(
    double left, double right, double bottom, double top, double near, double far) : CameraBase()
    , _Left(left), _Right(right), _Bottom(bottom), _Top(top), _Near(near), _Far(far)
    , _bIsPCalculated(false)
{
    _matProjection = Matrix4::S_IDENTITY_MATRIX;
    _VecPan = Vector2::S_ZERO_VECTOR;
}

OrthoCamera::~OrthoCamera()
{

}

void OrthoCamera::set_ortho(double left, double right, double bottom, double top, double near, double far0)
{
    _Left = left;
    _Right = right;
    _Bottom = bottom;
    _Top = top;
    _Near = near;
    _Far = far0;
    _bIsPCalculated = false;
}

void OrthoCamera::get_ortho(double& left, double& right, double& bottom, double& top, double& near, double& far0) const
{
    left= _Left ;
    right= _Right ;
    bottom= _Bottom ;
    top= _Top ;
    near= _Near ;
    far0= _Far  ;
}

Matrix4 OrthoCamera::get_projection_matrix()
{
    calculate_projection_matrix_i();
    return _matProjection;
}

Matrix4 OrthoCamera::get_view_projection_matrix()
{
    calculate_view_matrix_i();
    calculate_projection_matrix_i();
    return _matProjection*_mat_view;
}

void OrthoCamera::calculate_projection_matrix_i()
{
    if (!_bIsPCalculated)
    {
        _matProjection = Matrix4::S_IDENTITY_MATRIX;
        _matProjection[0][0] = 2.0f / ((_Right - _Left) * _ZoomFactor);
        _matProjection[1][1] = 2.0f / ((_Top - _Bottom) * _ZoomFactor);
        _matProjection[2][2] = -2.0f / (_Far - _Near);
        //这里是因为_Near和_Far是距离视点的距离（负的），即认为近远平面都在视点的观察反方向，即左下角点(left , bottom , -near)
        //右上角点(right , top , -far) 则 -far < -near 即 far > near
        _matProjection[3][0] = -(_Right + _Left + _VecPan.x*(_Right - _Left)) / (_Right - _Left);
        _matProjection[3][1] = -(_Top + _Bottom + _VecPan.y*(_Right - _Left)) / (_Top - _Bottom);
        _matProjection[3][2] = -(_Far + _Near) / (_Far - _Near);
        _bIsPCalculated = true;
    }
}

void OrthoCamera::zoom(double rate)
{
    //Check if rate is (-1 ~ 1)
    if (rate < 1.0 && rate > -1.0)
    {
        _ZoomFactor *= (1.0 + rate);
        _bIsPCalculated = false;
    }
}

void OrthoCamera::pan(const Vector2& pan)
{
    _VecPan += pan;
    //只能移动一半窗口
    //if (_VecPan.x > 1.0)
    //{
    //    _VecPan.x = 1.0;
    //}
    //if (_VecPan.x < -1.0)
    //{
    //    _VecPan.x = -1.0;
    //}
    //if (_VecPan.y > 1.0)
    //{
    //    _VecPan.y = 1.0;
    //}
    //if (_VecPan.y < -1.0)
    //{
    //    _VecPan.y = -1.0;
    //}
    _bIsPCalculated = false;
}

double OrthoCamera::get_near_clip_distance() const
{
    return _Near;
}

double OrthoCamera::get_far_clip_distance() const
{
    return _Far;
}

OrthoCamera& OrthoCamera::operator=(const OrthoCamera& camera)
{
    CameraBase::operator=(camera);

#define COPY_PARAMETER(p) this->p = camera.p
    COPY_PARAMETER(_Left);
    COPY_PARAMETER(_Right);
    COPY_PARAMETER(_Bottom);
    COPY_PARAMETER(_Top);
    COPY_PARAMETER(_Near);
    COPY_PARAMETER(_Far);
    COPY_PARAMETER(_matProjection);
    COPY_PARAMETER(_bIsPCalculated);
    COPY_PARAMETER(_ZoomFactor);
    COPY_PARAMETER(_VecPan);
#undef COPY_PARAMETER
    return *this;
}



MED_IMAGING_END_NAMESPACE

