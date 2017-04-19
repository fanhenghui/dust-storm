#include "mi_mouse_op_min_max_hint.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"

QMinMaxHintObject::QMinMaxHintObject(QObject* parent /*= 0*/):QObject(parent)
{

}

QMinMaxHintObject::~QMinMaxHintObject()
{

}

//void QMinMaxHintObject::Triggered(std::shared_ptr<MedImaging::SceneBase> pScene)
//{
//    emit triggered(pScene);
//}

void QMinMaxHintObject::Triggered(const std::string& s)
{
    emit triggered(s);
}

MED_IMAGING_BEGIN_NAMESPACE

MouseOpMinMaxHint::MouseOpMinMaxHint():m_pMinMaxHintObject(nullptr)
{

}

MouseOpMinMaxHint::~MouseOpMinMaxHint()
{

}

void MouseOpMinMaxHint::Press(const QPoint& pt)
{
    
}

void MouseOpMinMaxHint::Move(const QPoint& pt)
{

}

void MouseOpMinMaxHint::Release(const QPoint& pt)
{

}

void MouseOpMinMaxHint::DoubleClick(const QPoint& pt)
{
    if (m_pMinMaxHintObject)
    {
        m_pMinMaxHintObject->Triggered(m_pScene->GetName());
    }
}

void MouseOpMinMaxHint::SetMinMaxHintObject(QMinMaxHintObject* pObj)
{
    m_pMinMaxHintObject = pObj;
}

MED_IMAGING_END_NAMESPACE

    