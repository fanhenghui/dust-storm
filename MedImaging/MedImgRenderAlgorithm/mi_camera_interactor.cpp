#include "mi_camera_interactor.h"
#include "MedImgArithmetic/mi_quat4.h"
#include "MedImgArithmetic/mi_track_ball.h"
#include "mi_camera_calculator.h"

MED_IMAGING_BEGIN_NAMESPACE

OrthoCameraInteractor::OrthoCameraInteractor(std::shared_ptr<OrthoCamera> camera )
{
    set_initial_status(camera);
}

OrthoCameraInteractor::~OrthoCameraInteractor()
{ 

}

void OrthoCameraInteractor::set_initial_status(std::shared_ptr<OrthoCamera> camera)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(camera);
    _camera_init = *(camera.get());
    _camera = camera;
}

void OrthoCameraInteractor::reset_camera()
{
    *(_camera.get()) = _camera_init;
}

void OrthoCameraInteractor::resize(int width , int height)
{
    double left , right , bottom , top , far , near;
    _camera_init.get_ortho(left , right , bottom , top , near , far);

    const double ratio = (double)width / (double)height;
    if (ratio > 1.0)
    {
        double length = (right - left)*ratio;//Choose initial length
        _camera->get_ortho(left , right , bottom , top , near , far);
        double center = (right + left)*0.5;//Choose current center

        left = center - length*0.5;
        right = center + length*0.5;
    }
    else if (ratio < 1.0)
    {
        //Adjust bottom and top
        double length = (top - bottom)/ratio;//Choose initial length
        _camera->get_ortho(left , right , bottom , top , near , far);
        double center = (top + bottom)*0.5;//Choose current center

        bottom = center - length*0.5;
        top = center + length*0.5;
    }
    _camera->set_ortho(left , right , bottom , top , near , far);
}

void OrthoCameraInteractor::zoom(double scale)
{
    _camera->zoom(scale);
}

void OrthoCameraInteractor::zoom(const Point2& pre_pt , const Point2& cur_pt, int width, int height)
{
    const double scale = (cur_pt.y -pre_pt.y)/(double)height;
    this->zoom(scale);
}

void OrthoCameraInteractor::pan(const Vector2& pan)
{
    _camera->pan(pan);
}

void OrthoCameraInteractor::pan(const Point2& pre_pt , const Point2& cur_pt, int width, int height)
{
    Vector2 delta = (pre_pt - cur_pt);
    delta.x /= width;
    delta.y /= -height;
    delta *= 2.0;//归一化坐标是-1 ~ 1 需要乘以2
    this->pan(delta);
}

void OrthoCameraInteractor::rotate(const Matrix4& mat)
{
    _camera->rotate(mat);
}

void OrthoCameraInteractor::rotate(const Point2& pre_pt , const Point2& cur_pt, int width, int height)
{
    if (pre_pt == cur_pt)
    {
        return;
    }
    this->rotate(TrackBall::mouse_motion_to_rotation(pre_pt , cur_pt , width , height  , Point2::S_ZERO_POINT).to_matrix());
}







MED_IMAGING_END_NAMESPACE