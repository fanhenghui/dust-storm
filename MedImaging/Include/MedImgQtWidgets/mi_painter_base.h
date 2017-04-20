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
    PainterBase():m_pPainter(nullptr)
    {};

    virtual ~PainterBase() {};

    void set_painter(QPainter* pPainter) {m_pPainter = pPainter;};

    void set_scene(std::shared_ptr<SceneBase> pScene)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);
        m_pScene = pScene;
    }

    virtual void render() = 0;

protected:
    QPainter* m_pPainter;
    std::shared_ptr<SceneBase> m_pScene;
};

MED_IMAGING_END_NAMESPACE

#endif