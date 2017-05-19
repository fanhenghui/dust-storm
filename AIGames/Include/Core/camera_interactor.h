#ifndef MED_IMAGING_CAMERA_INTERACTOR_H
#define MED_IMAGING_CAMERA_INTERACTOR_H

#include <memory>
#include "Core/point2.h"
#include "Core/vector2.h"
#include "Core/matrix4.h"
#include "Core/ortho_camera.h"

class OrthoCamera;
class CameraCalculator;
class OrthoCameraInteractor
{
public:
    OrthoCameraInteractor(std::shared_ptr<OrthoCamera> camera );
    ~OrthoCameraInteractor();

    void reset_camera();
    void set_initial_status(std::shared_ptr<OrthoCamera> camera);

    void resize(int width , int height);
    void zoom(double scale);
    void pan(const Vector2& pan );
    void rotate(const Matrix4& mat);
    void zoom(const Point2& pre_pt , const Point2& cur_pt , int width, int height);
    void pan(const Point2& pre_pt , const Point2& cur_pt , int width, int height);
    void rotate(const Point2& pre_pt , const Point2& cur_pt , int width, int height);

protected:
private:
    OrthoCamera _camera_init;
    std::shared_ptr<OrthoCamera> _camera;
};

#endif