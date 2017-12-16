#ifndef MEDIMGRENDERALGO_CAMERA_INTERACTOR_H
#define MEDIMGRENDERALGO_CAMERA_INTERACTOR_H

#include "arithmetic/mi_matrix4.h"
#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_vector2.h"
#include "renderalgo/mi_render_algo_export.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class OrthoCamera;
class CameraCalculator;
class RenderAlgo_Export OrthoCameraInteractor {
public:
    explicit OrthoCameraInteractor(std::shared_ptr<OrthoCamera> camera);
    ~OrthoCameraInteractor();

    void reset_camera();
    void set_initial_status(std::shared_ptr<OrthoCamera> camera);

    void resize(int width, int height);
    void zoom(double scale);
    void pan(const Vector2& pan);
    void rotate(const Matrix4& mat);
    void zoom(const Point2& pre_pt, const Point2& cur_pt, int width, int height);
    void pan(const Point2& pre_pt, const Point2& cur_pt, int width, int height);
    void rotate(const Point2& pre_pt, const Point2& cur_pt, int width,
                int height);

protected:
private:
    OrthoCamera _camera_init;
    std::shared_ptr<OrthoCamera> _camera;
};

MED_IMG_END_NAMESPACE

#endif