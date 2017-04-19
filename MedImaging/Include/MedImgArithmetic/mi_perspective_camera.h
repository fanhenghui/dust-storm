#ifndef ARITHMETIC_PERSPECTIVE_CAMERA_H_
#define ARITHMETIC_PERSPECTIVE_CAMERA_H_

#include "MedImgArithmetic/mi_camera_base.h"
#include "MedImgArithmetic/mi_vector2.h"
#include "MedImgArithmetic/mi_matrix4.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export PerspectiveCamera : public CameraBase
{
public:
	PerspectiveCamera();

	PerspectiveCamera(double fovy, double aspect, double zNear, double zFar);

	virtual ~PerspectiveCamera();

	void SetPerspective(double fovy, double aspect, double zNear, double zFar);

	void SetNearClipDistance(double zNear);

	void SetFarClipDistance(double zFar);

	virtual double GetNearClipDistance() const;

	virtual double GetFarClipDistance() const;

	void SetFovy(double fovy);

	void SetAspectRatio(double aspect);

	virtual Matrix4 GetProjectionMatrix();

	virtual Matrix4 GetViewProjectionMatrix();

	virtual void Zoom(double rate);

	virtual void Pan(const Vector2& pan);

    PerspectiveCamera& operator =(const PerspectiveCamera& camera);

protected:
	virtual void CalculateProjectionMatrix_i();

private:
	double m_Fovy;
	double m_Aspect;
	double m_Near;
	double m_Far;
	Matrix4 m_matProjection;
	bool m_bIsPCalculated;
private:
};

MED_IMAGING_END_NAMESPACE
#endif