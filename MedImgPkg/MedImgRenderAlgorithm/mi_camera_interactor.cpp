#include "mi_camera_interactor.h"
#include "MedImgArithmetic/mi_quat4.h"
#include "MedImgArithmetic/mi_track_ball.h"

MED_IMG_BEGIN_NAMESPACE

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
        double left0 , right0 , bottom0 , top0 , far0 , near0;
        _camera->get_ortho(left0 , right0 , bottom0 , top0 , near0 , far0);
        double center = (right0 + left0)*0.5;//Choose current center

        left = center - length*0.5;
        right = center + length*0.5;
    }
    else if (ratio < 1.0)
    {
        //Adjust bottom and top
        double length = (top - bottom)/ratio;//Choose initial length
        double left0 , right0 , bottom0 , top0 , far0 , near0;
        _camera->get_ortho(left0 , right0 , bottom0 , top0 , near0 , far0);
        double center = (top0 + bottom0)*0.5;//Choose current center

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
    const Matrix4 mat_p = _camera->get_projection_matrix();
    if (mat_p.has_inverse())
    {
        const Matrix4 mat_vp = _camera->get_view_projection_matrix();
        const Point3 at_screen = mat_vp.transform(_camera->get_look_at());

        const double dx = (cur_pt.x - pre_pt.x)/width*2.0;
        const double dy = (pre_pt.y - cur_pt.y)/height*2.0;
        const Point3 pt = mat_p.get_inverse().transform(Point3(dx , dy , at_screen.z));

        this->pan(Vector2(pt.x , pt.y));
    }
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







MED_IMG_END_NAMESPACE