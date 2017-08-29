#include "mi_mouse_op_locate.h"
#include "renderalgo/mi_mpr_scene.h"
#include "arithmetic/mi_point2.h"
#include "mi_model_cross_hair.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpLocate::MouseOpLocate() {

}

MouseOpLocate::~MouseOpLocate() {

}

void MouseOpLocate::press(const QPointF& pt) {
    if (!_scene) {
        return;
    }

    QTWIDGETS_CHECK_NULL_EXCEPTION(_model);

    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(_scene);

    if (scene) {
        if (_model->locate(scene , Point2(pt.x() , pt.y()))) {
            _model->notify();
        }
    }

    _pre_point = pt;
}

void MouseOpLocate::move(const QPointF&) {
}

void MouseOpLocate::release(const QPointF&) {
}

void MouseOpLocate::double_click(const QPointF&) {
}

void MouseOpLocate::set_crosshair_model(std::shared_ptr<CrosshairModel> model) {
    _model = model;
}

void MouseOpLocate::wheel_slide(int) {

}

MED_IMG_END_NAMESPACE