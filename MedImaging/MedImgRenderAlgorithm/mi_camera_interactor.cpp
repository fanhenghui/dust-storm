#include "mi_camera_interactor.h"
#include "MedImgArithmetic/mi_quat4.h"
#include "MedImgArithmetic/mi_track_ball.h"
#include "mi_camera_calculator.h"

MED_IMAGING_BEGIN_NAMESPACE

OrthoCameraInteractor::OrthoCameraInteractor(std::shared_ptr<OrthoCamera> pCamera )
{
    SetInitialStatus(pCamera);
}

OrthoCameraInteractor::~OrthoCameraInteractor()
{ 

}

void OrthoCameraInteractor::SetInitialStatus(std::shared_ptr<OrthoCamera> pCamera)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(pCamera);
    m_CameraInitial = *(pCamera.get());
    m_pCamera = pCamera;
}

void OrthoCameraInteractor::ResetCamera()
{
    *(m_pCamera.get()) = m_CameraInitial;
}

void OrthoCameraInteractor::Resize(int iWidth , int iHeight)
{
    double dLeft , dRight , dBottom , dTop , dFar , dNear;
    m_CameraInitial.GetOrtho(dLeft , dRight , dBottom , dTop , dNear , dFar);

    const double dRatio = (double)iWidth / (double)iHeight;
    if (dRatio > 1.0)
    {
        double dLength = (dRight - dLeft)*dRatio;//Choose initial length
        m_pCamera->GetOrtho(dLeft , dRight , dBottom , dTop , dNear , dFar);
        double dCenter = (dRight + dLeft)*0.5;//Choose current center

        dLeft = dCenter - dLength*0.5;
        dRight = dCenter + dLength*0.5;
    }
    else if (dRatio < 1.0)
    {
        //Adjust bottom and top
        double dLength = (dTop - dBottom)/dRatio;//Choose initial length
        m_pCamera->GetOrtho(dLeft , dRight , dBottom , dTop , dNear , dFar);
        double dCenter = (dTop + dBottom)*0.5;//Choose current center

        dBottom = dCenter - dLength*0.5;
        dTop = dCenter + dLength*0.5;
    }
    m_pCamera->SetOrtho(dLeft , dRight , dBottom , dTop , dNear , dFar);
}

void OrthoCameraInteractor::Zoom(double dScale)
{
    m_pCamera->Zoom(dScale);
}

void OrthoCameraInteractor::Zoom(const Point2& ptPre , const Point2& ptCur, int iWidth, int iHeight)
{
    const double dScale = (ptCur.y -ptPre.y)/(double)iHeight;
    this->Zoom(dScale);
}

void OrthoCameraInteractor::Pan(const Vector2& vPan)
{
    m_pCamera->Pan(vPan);
}

void OrthoCameraInteractor::Pan(const Point2& ptPre , const Point2& ptCur, int iWidth, int iHeight)
{
    Vector2 vecT = (ptPre - ptCur);
    vecT.x /= iWidth;
    vecT.y /= -iHeight;
    vecT *= 2.0;//归一化坐标是-1 ~ 1 需要乘以2
    this->Pan(vecT);
}

void OrthoCameraInteractor::Rotate(const Matrix4& mat)
{
    m_pCamera->Rotate(mat);
}

void OrthoCameraInteractor::Rotate(const Point2& ptPre , const Point2& ptCur, int iWidth, int iHeight)
{
    if (ptPre == ptCur)
    {
        return;
    }
    this->Rotate(TrackBall::MouseMotionToRotation(ptPre , ptCur , iWidth , iHeight  , Point2(0,0)).ToMatrix());
}







MED_IMAGING_END_NAMESPACE