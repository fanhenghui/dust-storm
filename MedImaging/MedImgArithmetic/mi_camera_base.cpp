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

void CameraBase::SetEye(const Point3& ptEye)
{
    m_ptEye = ptEye;
    m_bIsVCalculated = false;
}

void CameraBase::SetLookAt(const Point3& ptCenter)
{
    m_ptAt = ptCenter;
    m_bIsVCalculated = false;
}

void CameraBase::SetUpDirection(const Vector3& vecUp)
{
    m_vUp = vecUp;
    m_bIsVCalculated = false;
}

Point3 CameraBase::GetEye() const
{
    return m_ptEye;
}

Point3 CameraBase::GetLookAt() const
{
    return m_ptAt;
}

Vector3 CameraBase::GetUpDirection() const
{
    return m_vUp;
}

Matrix4 CameraBase::GetViewMatrix()
{
    CalculateViewMatrix_i();
    return m_matView;
}

void CameraBase::CalculateViewMatrix_i()
{
    if (!m_bIsVCalculated)
    {
        Vector3 vZ = m_ptEye - m_ptAt;
        vZ.Normalize();
        Vector3 vX = m_vUp.CrossProduct(vZ);
        vX.Normalize();
        Vector3 vY = vZ.CrossProduct(vX);

        m_matView[0][0] = vX.x;
        m_matView[1][0] = vX.y;
        m_matView[2][0] = vX.z;
        m_matView[3][0] = -(vX.DotProduct(m_ptEye - Point3(0,0,0)));
        m_matView[0][1] = vY.x;
        m_matView[1][1] = vY.y;
        m_matView[2][1] = vY.z;
        m_matView[3][1] = -(vY.DotProduct(m_ptEye - Point3(0,0,0)));
        m_matView[0][2] = vZ.x;
        m_matView[1][2] = vZ.y;
        m_matView[2][2] = vZ.z;
        m_matView[3][2] = -(vZ.DotProduct(m_ptEye - Point3(0,0,0)));
        m_matView[0][3] = 0;
        m_matView[1][3] = 0;
        m_matView[2][3] = 0;
        m_matView[3][3] = 1;

        m_bIsVCalculated = true;
    }
}

void CameraBase::GetOpenGLViewMatrix(float(&fMat)[16])
{
    CalculateViewMatrix_i();
    for (int  i = 0 ; i<16 ; ++i)
    {
        fMat[i] = static_cast<float>(m_matView._m[i]);
    }
}

void CameraBase::Rotate(const Quat4& quat)
{
    Rotate(quat.ToMatrix());
}

void CameraBase::Rotate(const Matrix4& matRotate)
{
    Matrix4 matViewPre = GetViewMatrix();
    if (!matViewPre.HasInverse())
    {
        return;
    }
    else
    {
        matViewPre.Prepend(MakeTranslate(Point3(0,0,0) - matViewPre.Transform(m_ptAt)));
        matViewPre = matViewPre.GetInverse() * matRotate * matViewPre;
        m_ptEye = matViewPre.Transform(m_ptEye);
        m_vUp = matViewPre.Transform(m_vUp);

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

Vector3 CameraBase::GetViewDirection() const
{
    return (m_ptAt - m_ptEye).GetNormalize();
}

MED_IMAGING_END_NAMESPACE