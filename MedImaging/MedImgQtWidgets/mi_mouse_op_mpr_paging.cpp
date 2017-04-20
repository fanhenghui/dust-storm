#include "mi_mouse_op_mpr_paging.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgArithmetic/mi_point2.h"
#include "mi_model_cross_hair.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpMPRPaging::MouseOpMPRPaging()
{

}

MouseOpMPRPaging::~MouseOpMPRPaging()
{

}

void MouseOpMPRPaging::press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_ptPre = pt;
}

void MouseOpMPRPaging::move(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    std::shared_ptr<MPRScene>  pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);

    const int iStep = int(pt.y() - m_ptPre.y());

    if (m_pModel)
    {
        m_pModel->page(pScene , iStep);
    }
    else
    {
        pScene->page(iStep);
    }
    m_ptPre = pt;
}

void MouseOpMPRPaging::release(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpMPRPaging::double_click(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpMPRPaging::set_crosshair_model(std::shared_ptr<CrosshairModel> pModel)
{
    m_pModel = pModel;
}

MED_IMAGING_END_NAMESPACE