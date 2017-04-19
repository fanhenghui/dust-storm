#include "mi_ortho_camera.h"

MED_IMAGING_BEGIN_NAMESPACE

    OrthoCamera::OrthoCamera() : CameraBase()
    , m_bIsPCalculated(false)
    , m_Left(0), m_Right(0), m_Bottom(0), m_Top(0), m_Near(0), m_Far(0), m_ZoomFactor(1.0)
{
    m_matProjection = Matrix4::kIdentityMatrix;
    m_VecPan = Vector2::kZeroVector;
}

OrthoCamera::OrthoCamera(
    double left, double right, double bottom, double top, double near, double far) : CameraBase()
    , m_Left(left), m_Right(right), m_Bottom(bottom), m_Top(top), m_Near(near), m_Far(far)
    , m_bIsPCalculated(false)
{
    m_matProjection = Matrix4::kIdentityMatrix;
    m_VecPan = Vector2::kZeroVector;
}

OrthoCamera::~OrthoCamera()
{

}

void OrthoCamera::SetOrtho(double left, double right, double bottom, double top, double near, double far0)
{
    m_Left = left;
    m_Right = right;
    m_Bottom = bottom;
    m_Top = top;
    m_Near = near;
    m_Far = far0;
    m_bIsPCalculated = false;
}

void OrthoCamera::GetOrtho(double& left, double& right, double& bottom, double& top, double& near, double& far0) const
{
    left= m_Left ;
    right= m_Right ;
    bottom= m_Bottom ;
    top= m_Top ;
    near= m_Near ;
    far0= m_Far  ;
}

Matrix4 OrthoCamera::GetProjectionMatrix()
{
    CalculateProjectionMatrix_i();
    return m_matProjection;
}

Matrix4 OrthoCamera::GetViewProjectionMatrix()
{
    CalculateViewMatrix_i();
    CalculateProjectionMatrix_i();
    return m_matProjection*m_matView;
}

void OrthoCamera::CalculateProjectionMatrix_i()
{
    if (!m_bIsPCalculated)
    {
        m_matProjection = Matrix4::kIdentityMatrix;
        m_matProjection[0][0] = 2.0f / ((m_Right - m_Left) * m_ZoomFactor);
        m_matProjection[1][1] = 2.0f / ((m_Top - m_Bottom) * m_ZoomFactor);
        m_matProjection[2][2] = -2.0f / (m_Far - m_Near);
        //这里是因为m_Near和m_Far是距离视点的距离（负的），即认为近远平面都在视点的观察反方向，即左下角点(left , bottom , -near)
        //右上角点(right , top , -far) 则 -far < -near 即 far > near
        m_matProjection[3][0] = -(m_Right + m_Left + m_VecPan.x*(m_Right - m_Left)) / (m_Right - m_Left);
        m_matProjection[3][1] = -(m_Top + m_Bottom + m_VecPan.y*(m_Right - m_Left)) / (m_Top - m_Bottom);
        m_matProjection[3][2] = -(m_Far + m_Near) / (m_Far - m_Near);
        m_bIsPCalculated = true;
    }
}

void OrthoCamera::Zoom(double rate)
{
    //Check if rate is (-1 ~ 1)
    if (rate < 1.0 && rate > -1.0)
    {
        m_ZoomFactor *= (1.0 + rate);
        m_bIsPCalculated = false;
    }
}

void OrthoCamera::Pan(const Vector2& pan)
{
    m_VecPan += pan;
    //只能移动一半窗口
    //if (m_VecPan.x > 1.0)
    //{
    //    m_VecPan.x = 1.0;
    //}
    //if (m_VecPan.x < -1.0)
    //{
    //    m_VecPan.x = -1.0;
    //}
    //if (m_VecPan.y > 1.0)
    //{
    //    m_VecPan.y = 1.0;
    //}
    //if (m_VecPan.y < -1.0)
    //{
    //    m_VecPan.y = -1.0;
    //}
    m_bIsPCalculated = false;
}

double OrthoCamera::GetNearClipDistance() const
{
    return m_Near;
}

double OrthoCamera::GetFarClipDistance() const
{
    return m_Far;
}

OrthoCamera& OrthoCamera::operator=(const OrthoCamera& camera)
{
    CameraBase::operator=(camera);

#define COPY_PARAMETER(p) this->p = camera.p
    COPY_PARAMETER(m_Left);
    COPY_PARAMETER(m_Right);
    COPY_PARAMETER(m_Bottom);
    COPY_PARAMETER(m_Top);
    COPY_PARAMETER(m_Near);
    COPY_PARAMETER(m_Far);
    COPY_PARAMETER(m_matProjection);
    COPY_PARAMETER(m_bIsPCalculated);
    COPY_PARAMETER(m_ZoomFactor);
    COPY_PARAMETER(m_VecPan);
#undef COPY_PARAMETER
    return *this;
}



MED_IMAGING_END_NAMESPACE

