#ifndef MED_IMG_MOUSE_OPERATION_INTERFACE_H
#define MED_IMG_MOUSE_OPERATION_INTERFACE_H

#include "qtpackage/mi_qt_package_export.h"
#include "Qt/qpoint.h"

namespace medical_imaging
{
    class SceneBase;
}

MED_IMG_BEGIN_NAMESPACE

class QtPackage_Export IMouseOp
{
public:
    IMouseOp() {};
    virtual ~IMouseOp() {};

    void set_scene(std::shared_ptr<SceneBase> scene) 
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(scene);
        _scene = scene;
    };

    virtual void press(const QPointF& pt) = 0;
    virtual void move(const QPointF& pt) = 0;
    virtual void release(const QPointF& pt) = 0;
    virtual void double_click(const QPointF& pt) = 0;
    virtual void wheel_slide(int value) = 0;

protected:
    QPointF _pre_point;
    std::shared_ptr<medical_imaging::SceneBase> _scene;
};

MED_IMG_END_NAMESPACE

#endif