#ifndef ARITHMETIC_CAMERA_BASE_H_
#define ARITHMETIC_CAMERA_BASE_H_

#include "MedImgArithmetic/mi_vector2.h"
#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_quat4.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export CameraBase
{
public:
    CameraBase();

    virtual ~CameraBase();

    void SetEye(const Point3& ptEye);

    void SetLookAt(const Point3& ptCenter);

    void SetUpDirection(const Vector3& vecUp);

    Point3 GetEye() const;

    Point3 GetLookAt() const;

    Vector3 GetUpDirection() const;

    Vector3 GetViewDirection() const;

    Matrix4 GetViewMatrix();

    virtual Matrix4 GetProjectionMatrix() = 0;

    virtual Matrix4 GetViewProjectionMatrix() = 0;

    virtual double GetNearClipDistance() const = 0;

    virtual double GetFarClipDistance() const = 0;

    void GetOpenGLViewMatrix(float (&fMat)[16]);

    void Rotate(const Quat4& quat);

    void Rotate(const Matrix4& mat);

    virtual void Zoom(double rate) = 0;

    virtual void Pan(const Vector2& pan) = 0;

    CameraBase& operator =(const CameraBase& camera);

protected:
    void CalculateViewMatrix_i();

    virtual void CalculateProjectionMatrix_i() = 0;

protected:
    Point3 m_ptEye;
    Point3 m_ptAt;
    Vector3 m_vUp;
    Matrix4 m_matView;
    bool m_bIsVCalculated;

};

MED_IMAGING_END_NAMESPACE

#endif
