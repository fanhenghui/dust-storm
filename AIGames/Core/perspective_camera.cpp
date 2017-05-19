#include "perspective_camera.h"
#include <cmath>



PerspectiveCamera::PerspectiveCamera() :CameraBase(),
_Fovy(0), _Aspect(0), _Near(0), _Far(0), _bIsPCalculated(false)
{
	_matProjection = Matrix4::S_IDENTITY_MATRIX;
}

PerspectiveCamera::PerspectiveCamera(double fovy, double aspect, double zNear, double zFar) :CameraBase(),
_Fovy(fovy), _Aspect(aspect), _Near(zNear), _Far(zFar), _bIsPCalculated(false)
{
	_matProjection = Matrix4::S_IDENTITY_MATRIX;
}

PerspectiveCamera::~PerspectiveCamera()
{

}

void PerspectiveCamera::set_perspective(double fovy, double aspect, double zNear, double zFar)
{
	_Fovy = fovy;
	_Aspect = aspect;
	_Near = zNear;
	_Far = zFar;
	_bIsPCalculated = false;
}

void PerspectiveCamera::set_near_clip_distance(double zNear)
{
	_Near = zNear;
	_bIsPCalculated = false;
}

void PerspectiveCamera::set_far_clip_distance(double zFar)
{
	_Far = zFar;
	_bIsPCalculated = false;
}

void PerspectiveCamera::set_fovy(double fovy)
{
	_Fovy = fovy;
	_bIsPCalculated = false;
}

void PerspectiveCamera::set_aspect_ratio(double aspect)
{
	_Aspect = aspect;
	_bIsPCalculated = false;
}

Matrix4 PerspectiveCamera::get_projection_matrix()
{
	calculate_projection_matrix_i();
	return _matProjection;
}

Matrix4 PerspectiveCamera::get_view_projection_matrix()
{
	calculate_view_matrix_i();
	calculate_projection_matrix_i();
	return _matProjection*_mat_view;
}

void PerspectiveCamera::calculate_projection_matrix_i()
{
	if (!_bIsPCalculated)
	{
		double range = tan(_Fovy / 2.0f) * _Near;
		double left = -range * _Aspect;
		double right = range * _Aspect;
		double bottom = -range;
		double top = range;

		_matProjection = Matrix4::S_IDENTITY_MATRIX;
		_matProjection[0][0] = (2.0f * _Near) / (right - left);
		_matProjection[1][1] = (2.0f* _Near) / (top - bottom);
		_matProjection[2][2] = -(_Far + _Near) / (_Far - _Near);
		_matProjection[2][3] = -1.0f;
		_matProjection[3][2] = -(2.0f* _Far * _Near) / (_Far - _Near);

		_bIsPCalculated = true;
	}
}

void PerspectiveCamera::zoom(double rate)
{
	rate;
}

void PerspectiveCamera::pan(const Vector2& pan)
{
	pan;
}

double PerspectiveCamera::get_near_clip_distance() const
{
	return _Near;
}

double PerspectiveCamera::get_far_clip_distance() const
{
	return _Far;
}

PerspectiveCamera& PerspectiveCamera::operator=(const PerspectiveCamera& camera)
{
#define COPY_PARAMETER(p) this->p = camera.p
    COPY_PARAMETER(_Fovy);
    COPY_PARAMETER(_Aspect);
    COPY_PARAMETER(_Near);
    COPY_PARAMETER(_Far);
    COPY_PARAMETER(_matProjection);
    COPY_PARAMETER(_bIsPCalculated);
#undef COPY_PARAMETER
    return *this;
}

