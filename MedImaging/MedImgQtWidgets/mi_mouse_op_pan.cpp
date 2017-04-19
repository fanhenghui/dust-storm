#include "mi_mouse_op_pan.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgArithmetic/mi_point2.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpPan::MouseOpPan()
{

}

MouseOpPan::~MouseOpPan()
{

}

void MouseOpPan::Press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_ptPre = pt;
}

void MouseOpPan::Move(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_pScene->Pan(Point2(m_ptPre.x() , m_ptPre.y()) , Point2(pt.x() , pt.y()));
    m_ptPre = pt;
}

void MouseOpPan::Release(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpPan::DoubleClick(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

MED_IMAGING_END_NAMESPACE