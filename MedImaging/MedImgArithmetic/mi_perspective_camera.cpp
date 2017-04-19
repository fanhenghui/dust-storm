#include "mi_perspective_camera.h"
#include <cmath>

MED_IMAGING_BEGIN_NAMESPACE

PerspectiveCamera::PerspectiveCamera() :CameraBase(),
m_Fovy(0), m_Aspect(0), m_Near(0), m_Far(0), m_bIsPCalculated(false)
{
	m_matProjection = Matrix4::kIdentityMatrix;
}

PerspectiveCamera::PerspectiveCamera(double fovy, double aspect, double zNear, double zFar) :CameraBase(),
m_Fovy(fovy), m_Aspect(aspect), m_Near(zNear), m_Far(zFar), m_bIsPCalculated(false)
{
	m_matProjection = Matrix4::kIdentityMatrix;
}

PerspectiveCamera::~PerspectiveCamera()
{

}

void PerspectiveCamera::SetPerspective(double fovy, double aspect, double zNear, double zFar)
{
	m_Fovy = fovy;
	m_Aspect = aspect;
	m_Near = zNear;
	m_Far = zFar;
	m_bIsPCalculated = false;
}

void PerspectiveCamera::SetNearClipDistance(double zNear)
{
	m_Near = zNear;
	m_bIsPCalculated = false;
}

void PerspectiveCamera::SetFarClipDistance(double zFar)
{
	m_Far = zFar;
	m_bIsPCalculated = false;
}

void PerspectiveCamera::SetFovy(double fovy)
{
	m_Fovy = fovy;
	m_bIsPCalculated = false;
}

void PerspectiveCamera::SetAspectRatio(double aspect)
{
	m_Aspect = aspect;
	m_bIsPCalculated = false;
}

Matrix4 PerspectiveCamera::GetProjectionMatrix()
{
	CalculateProjectionMatrix_i();
	return m_matProjection;
}

Matrix4 PerspectiveCamera::GetViewProjectionMatrix()
{
	CalculateViewMatrix_i();
	CalculateProjectionMatrix_i();
	return m_matProjection*m_matView;
}

void PerspectiveCamera::CalculateProjectionMatrix_i()
{
	if (!m_bIsPCalculated)
	{
		double range = tan(m_Fovy / 2.0f) * m_Near;
		double left = -range * m_Aspect;
		double right = range * m_Aspect;
		double bottom = -range;
		double top = range;

		m_matProjection = Matrix4::kIdentityMatrix;
		m_matProjection[0][0] = (2.0f * m_Near) / (right - left);
		m_matProjection[1][1] = (2.0f* m_Near) / (top - bottom);
		m_matProjection[2][2] = -(m_Far + m_Near) / (m_Far - m_Near);
		m_matProjection[2][3] = -1.0f;
		m_matProjection[3][2] = -(2.0f* m_Far * m_Near) / (m_Far - m_Near);

		m_bIsPCalculated = true;
	}
}

void PerspectiveCamera::Zoom(double rate)
{
	rate;
}

void PerspectiveCamera::Pan(const Vector2& pan)
{
	pan;
}

double PerspectiveCamera::GetNearClipDistance() const
{
	return m_Near;
}

double PerspectiveCamera::GetFarClipDistance() const
{
	return m_Far;
}

PerspectiveCamera& PerspectiveCamera::operator=(const PerspectiveCamera& camera)
{
#define COPY_PARAMETER(p) this->p = camera.p
    COPY_PARAMETER(m_Fovy);
    COPY_PARAMETER(m_Aspect);
    COPY_PARAMETER(m_Near);
    COPY_PARAMETER(m_Far);
    COPY_PARAMETER(m_matProjection);
    COPY_PARAMETER(m_bIsPCalculated);
#undef COPY_PARAMETER
    return *this;
}

MED_IMAGING_END_NAMESPACE