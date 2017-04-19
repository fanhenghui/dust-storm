#ifndef MED_IMAGING_CAMERA_INTERACTOR_H
#define MED_IMAGING_CAMERA_INTERACTOR_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_vector2.h"
#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_ortho_camera.h"

MED_IMAGING_BEGIN_NAMESPACE

class OrthoCamera;
class CameraCalculator;
class RenderAlgo_Export OrthoCameraInteractor
{
public:
    OrthoCameraInteractor(std::shared_ptr<OrthoCamera> pCamera );
    ~OrthoCameraInteractor();
    void ResetCamera();
    void SetInitialStatus(std::shared_ptr<OrthoCamera> pCamera);
    void Resize(int iWidth , int iHeight);
    void Zoom(double dScale);
    void Pan(const Vector2& vPan );
    void Rotate(const Matrix4& mat);
    void Zoom(const Point2& ptPre , const Point2& ptCur , int iWidth, int iHeight);
    void Pan(const Point2& ptPre , const Point2& ptCur , int iWidth, int iHeight);
    void Rotate(const Point2& ptPre , const Point2& ptCur , int iWidth, int iHeight);

protected:
private:
    OrthoCamera m_CameraInitial;
    std::shared_ptr<OrthoCamera> m_pCamera;
};

MED_IMAGING_END_NAMESPACE

#endif