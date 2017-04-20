#include "mi_camera_base.h"

MED_IMAGING_BEGIN_NAMESPACE

    CameraBase::CameraBase()
{
    m_ptEye = Point3::kZeroPoint;
    m_ptAt = Point3::kZeroPoint;
    m_vUp = Vector3::kZeroVector;
    m_matView = Matrix4::kIdentityMatrix;
    m_bIsVCalculated = false;
}

CameraBase::~CameraBase()
{
}

void CameraBase::set_eye(const Point3& ptEye)
{
    m_ptEye = ptEye;
    m_bIsVCalculated = false;
}

void CameraBase::set_look_at(const Point3& ptCenter)
{
    m_ptAt = ptCenter;
    m_bIsVCalculated = false;
}

void CameraBase::set_up_direction(const Vector3& vecUp)
{
    m_vUp = vecUp;
    m_bIsVCalculated = false;
}

Point3 CameraBase::get_eye() const
{
    return m_ptEye;
}

Point3 CameraBase::get_look_at() const
{
    return m_ptAt;
}

Vector3 CameraBase::get_up_direction() const
{
    return m_vUp;
}

Matrix4 CameraBase::get_view_matrix()
{
    calculate_view_matrix_i();
    return m_matView;
}

void CameraBase::calculate_view_matrix_i()
{
    if (!m_bIsVCalculated)
    {
        Vector3 vZ = m_ptEye - m_ptAt;
        vZ.normalize();
        Vector3 vX = m_vUp.cross_product(vZ);
        vX.normalize();
        Vector3 vY = vZ.cross_product(vX);

        m_matView[0][0] = vX.x;
        m_matView[1][0] = vX.y;
        m_matView[2][0] = vX.z;
        m_matView[3][0] = -(vX.dot_product(m_ptEye - Point3(0,0,0)));
        m_matView[0][1] = vY.x;
        m_matView[1][1] = vY.y;
        m_matView[2][1] = vY.z;
        m_matView[3][1] = -(vY.dot_product(m_ptEye - Point3(0,0,0)));
        m_matView[0][2] = vZ.x;
        m_matView[1][2] = vZ.y;
        m_matView[2][2] = vZ.z;
        m_matView[3][2] = -(vZ.dot_product(m_ptEye - Point3(0,0,0)));
        m_matView[0][3] = 0;
        m_matView[1][3] = 0;
        m_matView[2][3] = 0;
        m_matView[3][3] = 1;

        m_bIsVCalculated = true;
    }
}

void CameraBase::rotate(const Quat4& quat)
{
    rotate(quat.to_matrix());
}

void CameraBase::rotate(const Matrix4& matRotate)
{
    Matrix4 matViewPre = get_view_matrix();
    if (!matViewPre.has_inverse())
    {
        return;
    }
    else
    {
        matViewPre.prepend(make_translate(Point3(0,0,0) - matViewPre.transform(m_ptAt)));
        matViewPre = matViewPre.get_inverse() * matRotate * matViewPre;
        m_ptEye = matViewPre.transform(m_ptEye);
        m_vUp = matViewPre.transform(m_vUp);

        m_bIsVCalculated = false;
    }

}

CameraBase& CameraBase::operator=(const CameraBase& camera)
{
#define COPY_PARAMETER(p) this->p = camera.p
    COPY_PARAMETER(m_ptEye);
    COPY_PARAMETER(m_ptAt);
    COPY_PARAMETER(m_vUp);
    COPY_PARAMETER(m_matView);
    COPY_PARAMETER(m_bIsVCalculated);
#undef COPY_PARAMETER
    return *this;
}

Vector3 CameraBase::get_view_direction() const
{
    return (m_ptAt - m_ptEye).get_normalize();
}

MED_IMAGING_END_NAMESPACE