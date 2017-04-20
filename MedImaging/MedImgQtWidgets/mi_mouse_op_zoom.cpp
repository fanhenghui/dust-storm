#include "mi_mouse_op_zoom.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgArithmetic/mi_point2.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpZoom::MouseOpZoom()
{

}

MouseOpZoom::~MouseOpZoom()
{

}

void MouseOpZoom::press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_ptPre = pt;
}

void MouseOpZoom::move(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_pScene->zoom(Point2(m_ptPre.x() , m_ptPre.y()) , Point2(pt.x() , pt.y()));
    m_ptPre = pt;
}

void MouseOpZoom::release(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpZoom::double_click(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

MED_IMAGING_END_NAMESPACE