#ifndef MED_IMAGING_MOUSE_OPERATION_INTERFACE_H
#define MED_IMAGING_MOUSE_OPERATION_INTERFACE_H

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "Qt/qpoint.h"

namespace medical_imaging
{
    class SceneBase;
}

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export IMouseOp
{
public:
    IMouseOp() {};
    virtual ~IMouseOp() {};
    void set_scene(std::shared_ptr<SceneBase> pScene) 
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);
        m_pScene = pScene;
    };
    virtual void press(const QPoint& pt) = 0;
    virtual void move(const QPoint& pt) = 0;
    virtual void release(const QPoint& pt) = 0;
    virtual void double_click(const QPoint& pt) = 0;
protected:
    QPoint m_ptPre;
    std::shared_ptr<medical_imaging::SceneBase> m_pScene;
};

MED_IMAGING_END_NAMESPACE

#endif