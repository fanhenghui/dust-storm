#ifndef MED_IMAGING_MOUSE_OPERATION_INTERFACE_H
#define MED_IMAGING_MOUSE_OPERATION_INTERFACE_H

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "Qt/qpoint.h"

namespace MedImaging
{
    class SceneBase;
}

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export IMouseOp
{
public:
    IMouseOp() {};
    virtual ~IMouseOp() {};
    void SetScene(std::shared_ptr<SceneBase> pScene) 
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);
        m_pScene = pScene;
    };
    virtual void Press(const QPoint& pt) = 0;
    virtual void Move(const QPoint& pt) = 0;
    virtual void Release(const QPoint& pt) = 0;
    virtual void DoubleClick(const QPoint& pt) = 0;
protected:
    QPoint m_ptPre;
    std::shared_ptr<MedImaging::SceneBase> m_pScene;
};

MED_IMAGING_END_NAMESPACE

#endif