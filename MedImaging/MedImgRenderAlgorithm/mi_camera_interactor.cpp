#include "mi_camera_interactor.h"
#include "MedImgArithmetic/mi_quat4.h"
#include "MedImgArithmetic/mi_track_ball.h"
#include "mi_camera_calculator.h"

MED_IMAGING_BEGIN_NAMESPACE

OrthoCameraInteractor::OrthoCameraInteractor(std::shared_ptr<OrthoCamera> pCamera )
{
    set_initial_status(pCamera);
}

OrthoCameraInteractor::~OrthoCameraInteractor()
{ 

}

void OrthoCameraInteractor::set_initial_status(std::shared_ptr<OrthoCamera> pCamera)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(pCamera);
    m_CameraInitial = *(pCamera.get());
    m_pCamera = pCamera;
}

void OrthoCameraInteractor::reset_camera()
{
    *(m_pCamera.get()) = m_CameraInitial;
}

void OrthoCameraInteractor::resize(int iWidth , int iHeight)
{
    double dLeft , dRight , dBottom , dTop , dFar , dNear;
    m_CameraInitial.get_ortho(dLeft , dRight , dBottom , dTop , dNear , dFar);

    const double dRatio = (double)iWidth / (double)iHeight;
    if (dRatio > 1.0)
    {
        double dLength = (dRight - dLeft)*dRatio;//Choose initial length
        m_pCamera->get_ortho(dLeft , dRight , dBottom , dTop , dNear , dFar);
        double dCenter = (dRight + dLeft)*0.5;//Choose current center

        dLeft = dCenter - dLength*0.5;
        dRight = dCenter + dLength*0.5;
    }
    else if (dRatio < 1.0)
    {
        //Adjust bottom and top
        double dLength = (dTop - dBottom)/dRatio;//Choose initial length
        m_pCamera->get_ortho(dLeft , dRight , dBottom , dTop , dNear , dFar);
        double dCenter = (dTop + dBottom)*0.5;//Choose current center

        dBottom = dCenter - dLength*0.5;
        dTop = dCenter + dLength*0.5;
    }
    m_pCamera->set_ortho(dLeft , dRight , dBottom , dTop , dNear , dFar);
}

void OrthoCameraInteractor::zoom(double dScale)
{
    m_pCamera->zoom(dScale);
}

void OrthoCameraInteractor::zoom(const Point2& ptPre , const Point2& ptCur, int iWidth, int iHeight)
{
    const double dScale = (ptCur.y -ptPre.y)/(double)iHeight;
    this->zoom(dScale);
}

void OrthoCameraInteractor::pan(const Vector2& vPan)
{
    m_pCamera->pan(vPan);
}

void OrthoCameraInteractor::pan(const Point2& ptPre , const Point2& ptCur, int iWidth, int iHeight)
{
    Vector2 vecT = (ptPre - ptCur);
    vecT.x /= iWidth;
    vecT.y /= -iHeight;
    vecT *= 2.0;//归一化坐标是-1 ~ 1 需要乘以2
    this->pan(vecT);
}

void OrthoCameraInteractor::rotate(const Matrix4& mat)
{
    m_pCamera->rotate(mat);
}

void OrthoCameraInteractor::rotate(const Point2& ptPre , const Point2& ptCur, int iWidth, int iHeight)
{
    if (ptPre == ptCur)
    {
        return;
    }
    this->rotate(TrackBall::mouse_motion_to_rotation(ptPre , ptCur , iWidth , iHeight  , Point2(0,0)).to_matrix());
}







MED_IMAGING_END_NAMESPACE