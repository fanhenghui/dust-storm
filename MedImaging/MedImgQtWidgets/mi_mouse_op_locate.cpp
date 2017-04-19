#include "mi_mouse_op_locate.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgArithmetic/mi_point2.h"
#include "mi_model_cross_hair.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpLocate::MouseOpLocate()
{

}

MouseOpLocate::~MouseOpLocate()
{

}

void MouseOpLocate::Press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    QTWIDGETS_CHECK_NULL_EXCEPTION(m_pModel);

    std::shared_ptr<MPRScene>  pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
    if (pScene)
    {
        if(m_pModel->Locate(pScene , Point2(pt.x() , pt.y()) ))
        {
            m_pModel->NotifyAllObserver();
        }
    }

    m_ptPre = pt;
}

void MouseOpLocate::Move(const QPoint& )
{
}

void MouseOpLocate::Release(const QPoint& )
{
}

void MouseOpLocate::DoubleClick(const QPoint& )
{
}

void MouseOpLocate::SetCrosshairModel(std::shared_ptr<CrosshairModel> pModel)
{
    m_pModel= pModel;
}

MED_IMAGING_END_NAMESPACE