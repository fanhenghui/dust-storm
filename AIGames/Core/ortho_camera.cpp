#include "ortho_camera.h"



    OrthoCamera::OrthoCamera() : CameraBase()
    , _is_proj_mat_cal(false)
    , _left(0), _right(0), _bottom(0), _top(0), _near(0), _far(0), _ZoomFactor(1.0)
{
    _mat_projection = Matrix4::S_IDENTITY_MATRIX;
    _VecPan = Vector2::S_ZERO_VECTOR;
}

OrthoCamera::OrthoCamera(
    double left, double right, double bottom, double top, double near, double far) : CameraBase()
    , _left(left), _right(right), _bottom(bottom), _top(top), _near(near), _far(far)
    , _is_proj_mat_cal(false)
{
    _mat_projection = Matrix4::S_IDENTITY_MATRIX;
    _VecPan = Vector2::S_ZERO_VECTOR;
}

OrthoCamera::~OrthoCamera()
{

}

void OrthoCamera::set_ortho(double left, double right, double bottom, double top, double near, double far0)
{
    _left = left;
    _right = right;
    _bottom = bottom;
    _top = top;
    _near = near;
    _far = far0;
    _is_proj_mat_cal = false;
}

void OrthoCamera::get_ortho(double& left, double& right, double& bottom, double& top, double& near, double& far0) const
{
    left= _left ;
    right= _right ;
    bottom= _bottom ;
    top= _top ;
    near= _near ;
    far0= _far  ;
}

Matrix4 OrthoCamera::get_projection_matrix()
{
    calculate_projection_matrix_i();
    return _mat_projection;
}

Matrix4 OrthoCamera::get_view_projection_matrix()
{
    calculate_view_matrix_i();
    calculate_projection_matrix_i();
    return _mat_projection*_mat_view;
}

void OrthoCamera::calculate_projection_matrix_i()
{
    if (!_is_proj_mat_cal)
    {
        _mat_projection = Matrix4::S_IDENTITY_MATRIX;
        _mat_projection[0][0] = 2.0f / ((_right - _left) * _ZoomFactor);
        _mat_projection[1][1] = 2.0f / ((_top - _bottom) * _ZoomFactor);
        _mat_projection[2][2] = -2.0f / (_far - _near);
        //这里是因为_Near和_Far是距离视点的距离（负的），即认为近远平面都在视点的观察反方向，即左下角点(left , bottom , -near)
        //右上角点(right , top , -far) 则 -far < -near 即 far > near
        _mat_projection[3][0] = -(_right + _left + _VecPan.x*(_right - _left)) / (_right - _left);
        _mat_projection[3][1] = -(_top + _bottom + _VecPan.y*(_right - _left)) / (_top - _bottom);
        _mat_projection[3][2] = -(_far + _near) / (_far - _near);
        _is_proj_mat_cal = true;
    }
}

void OrthoCamera::zoom(double rate)
{
    //Check if rate is (-1 ~ 1)
    if (rate < 1.0 && rate > -1.0)
    {
        _ZoomFactor *= (1.0 + rate);
        _is_proj_mat_cal = false;
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
    _is_proj_mat_cal = false;
}

double OrthoCamera::get_near_clip_distance() const
{
    return _near;
}

double OrthoCamera::get_far_clip_distance() const
{
    return _far;
}

OrthoCamera& OrthoCamera::operator=(const OrthoCamera& camera)
{
    CameraBase::operator=(camera);

#define COPY_PARAMETER(p) this->p = camera.p
    COPY_PARAMETER(_left);
    COPY_PARAMETER(_right);
    COPY_PARAMETER(_bottom);
    COPY_PARAMETER(_top);
    COPY_PARAMETER(_near);
    COPY_PARAMETER(_far);
    COPY_PARAMETER(_mat_projection);
    COPY_PARAMETER(_is_proj_mat_cal);
    COPY_PARAMETER(_ZoomFactor);
    COPY_PARAMETER(_VecPan);
#undef COPY_PARAMETER
    return *this;
}

bool OrthoCamera::operator==(const OrthoCamera& camera) const
{
    if (_is_proj_mat_cal && _is_view_mat_cal && camera._is_proj_mat_cal && camera._is_view_mat_cal)
    {
        return _mat_view == camera._mat_view && _mat_projection == camera._mat_projection;
    }
    else
    {
        return false;
    }
}





