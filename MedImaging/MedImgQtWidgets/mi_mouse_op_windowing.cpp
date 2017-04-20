#include "mi_mouse_op_windowing.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgRenderAlgorithm/mi_ray_cast_scene.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpWindowing::MouseOpWindowing()
{

}

MouseOpWindowing::~MouseOpWindowing()
{

}

void MouseOpWindowing::press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_ptPre = pt;
}

void MouseOpWindowing::move(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    //TODO MPR VR diverse strategy
    std::shared_ptr<RayCastScene>  pScene = std::dynamic_pointer_cast<RayCastScene>(m_pScene);
    if (pScene)
    {
        float fWW , fWL;
        pScene->get_global_window_level(fWW, fWL);
        float fDeltaWW = pt.x() - m_ptPre.x();
        float fDeltaWL = m_ptPre.y() - pt.y();
        if (fWW + fDeltaWW > 1.0f)
        {
            fWW += fDeltaWW;
        }
        fWL += fDeltaWL;
        pScene->set_global_window_level(fWW , fWL);
    }

    m_ptPre = pt;
}

void MouseOpWindowing::release(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpWindowing::double_click(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

MED_IMAGING_END_NAMESPACE