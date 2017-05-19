#include "camera_base.h"



    CameraBase::CameraBase()
{
    _eye = Point3::S_ZERO_POINT;
    _at = Point3::S_ZERO_POINT;
    _up = Vector3::S_ZERO_VECTOR;
    _mat_view = Matrix4::S_IDENTITY_MATRIX;
    _is_view_mat_cal = false;
}

CameraBase::~CameraBase()
{
}

void CameraBase::set_eye(const Point3& eye)
{
    _eye = eye;
    _is_view_mat_cal = false;
}

void CameraBase::set_look_at(const Point3& center)
{
    _at = center;
    _is_view_mat_cal = false;
}

void CameraBase::set_up_direction(const Vector3& up)
{
    _up = up;
    _is_view_mat_cal = false;
}

Point3 CameraBase::get_eye() const
{
    return _eye;
}

Point3 CameraBase::get_look_at() const
{
    return _at;
}

Vector3 CameraBase::get_up_direction() const
{
    return _up;
}

Matrix4 CameraBase::get_view_matrix()
{
    calculate_view_matrix_i();
    return _mat_view;
}

void CameraBase::calculate_view_matrix_i()
{
    if (!_is_view_mat_cal)
    {
        Vector3 v_z = _eye - _at;
        v_z.normalize();
        Vector3 v_x = _up.cross_product(v_z);
        v_x.normalize();
        Vector3 v_y = v_z.cross_product(v_x);

        _mat_view[0][0] = v_x.x;
        _mat_view[1][0] = v_x.y;
        _mat_view[2][0] = v_x.z;
        _mat_view[3][0] = -(v_x.dot_product(_eye - Point3(0,0,0)));
        _mat_view[0][1] = v_y.x;
        _mat_view[1][1] = v_y.y;
        _mat_view[2][1] = v_y.z;
        _mat_view[3][1] = -(v_y.dot_product(_eye - Point3(0,0,0)));
        _mat_view[0][2] = v_z.x;
        _mat_view[1][2] = v_z.y;
        _mat_view[2][2] = v_z.z;
        _mat_view[3][2] = -(v_z.dot_product(_eye - Point3(0,0,0)));
        _mat_view[0][3] = 0;
        _mat_view[1][3] = 0;
        _mat_view[2][3] = 0;
        _mat_view[3][3] = 1;

        _is_view_mat_cal = true;
    }
}

void CameraBase::rotate(const Quat4& quat)
{
    rotate(quat.to_matrix());
}

void CameraBase::rotate(const Matrix4& mat)
{
    Matrix4 mat_pre_view = get_view_matrix();
    if (!mat_pre_view.has_inverse())
    {
        return;
    }
    else
    {
        mat_pre_view.prepend(make_translate(Point3(0,0,0) - mat_pre_view.transform(_at)));
        mat_pre_view = mat_pre_view.get_inverse() * mat * mat_pre_view;
        _eye = mat_pre_view.transform(_eye);
        _up = mat_pre_view.transform(_up);

        _is_view_mat_cal = false;
    }

}

CameraBase& CameraBase::operator=(const CameraBase& camera)
{
#define COPY_PARAMETER(p) this->p = camera.p
    COPY_PARAMETER(_eye);
    COPY_PARAMETER(_at);
    COPY_PARAMETER(_up);
    COPY_PARAMETER(_mat_view);
    COPY_PARAMETER(_is_view_mat_cal);
#undef COPY_PARAMETER
    return *this;
}

Vector3 CameraBase::get_view_direction() const
{
    return (_at - _eye).get_normalize();
}

bool CameraBase::operator==(const CameraBase& camera) const
{
    if (_is_view_mat_cal && camera._is_view_mat_cal)
    {
        return _mat_view == camera._mat_view;
    }
    else
    {
        return false;
    }
}

