#ifndef MED_IMAGING_PAINTER_H_
#define MED_IMAGING_PAINTER_H_

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "qt/qpainter.h"

class QPainter;

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export PainterBase
{
public:
    PainterBase():_painter(nullptr)
    {};

    virtual ~PainterBase() {};

    void set_painter(QPainter* painter) {_painter = painter;};

    void set_scene(std::shared_ptr<SceneBase> scene)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(scene);
        _scene = scene;
    }

    virtual void render() = 0;

protected:
    QPainter* _painter;
    std::shared_ptr<SceneBase> _scene;
};

MED_IMAGING_END_NAMESPACE

#endif