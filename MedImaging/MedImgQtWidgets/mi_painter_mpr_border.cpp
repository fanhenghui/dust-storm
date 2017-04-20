#include "mi_painter_mpr_border.h"

#include "mi_model_cross_hair.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

MED_IMAGING_BEGIN_NAMESPACE

MPRBorderPainter::MPRBorderPainter()
{

}

MPRBorderPainter::~MPRBorderPainter()
{

}

void MPRBorderPainter::Render()
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_pModel);
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_pScene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_pPainter);

    std::shared_ptr<MPRScene> pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);

    int iWidth(1) , iHeight(1);
    pScene->GetDisplaySize(iWidth , iHeight);

    RGBUnit color = m_pModel->GetBorderColor(pScene);

    if (m_pModel->CheckFocus(pScene))
    {
        QPen pen(QColor(0,255,255));
        pen.setWidth(7);
        m_pPainter->setPen(pen);
        m_pPainter->drawRect(QRect(0 , 0 , iWidth, iHeight));
    }
    else
    {
        QPen pen(QColor(color.r ,color.g , color.b));
        pen.setWidth(4);
        m_pPainter->setPen(pen);
        m_pPainter->drawRect(QRect(0 , 0 , iWidth, iHeight));
    }
}

void MPRBorderPainter::SetCrossHairModel(std::shared_ptr<CrosshairModel> pModel)
{
    m_pModel = pModel;
}

MED_IMAGING_END_NAMESPACE