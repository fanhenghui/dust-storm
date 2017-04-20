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
    void reset_camera();
    void set_initial_status(std::shared_ptr<OrthoCamera> pCamera);
    void resize(int iWidth , int iHeight);
    void zoom(double dScale);
    void pan(const Vector2& vPan );
    void rotate(const Matrix4& mat);
    void zoom(const Point2& ptPre , const Point2& ptCur , int iWidth, int iHeight);
    void pan(const Point2& ptPre , const Point2& ptCur , int iWidth, int iHeight);
    void rotate(const Point2& ptPre , const Point2& ptCur , int iWidth, int iHeight);

protected:
private:
    OrthoCamera m_CameraInitial;
    std::shared_ptr<OrthoCamera> m_pCamera;
};

MED_IMAGING_END_NAMESPACE

#endif