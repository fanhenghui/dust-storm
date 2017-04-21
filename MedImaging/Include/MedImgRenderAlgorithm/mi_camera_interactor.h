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

MED_IMAGING_END_NAMESPACE

#endif