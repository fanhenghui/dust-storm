#include "mi_mouse_op_rotate.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgArithmetic/mi_point2.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpRotate::MouseOpRotate()
{

}

MouseOpRotate::~MouseOpRotate()
{

}

void MouseOpRotate::Press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_ptPre = pt;
}

void MouseOpRotate::Move(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_pScene->Rotate(Point2(m_ptPre.x() , m_ptPre.y()) , Point2(pt.x() , pt.y()));
    m_ptPre = pt;
}

void MouseOpRotate::Release(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpRotate::DoubleClick(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

MED_IMAGING_END_NAMESPACE
