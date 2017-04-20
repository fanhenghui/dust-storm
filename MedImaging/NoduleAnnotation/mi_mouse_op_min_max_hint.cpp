#include "mi_mouse_op_min_max_hint.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"

QMinMaxHintObject::QMinMaxHintObject(QObject* parent /*= 0*/):QObject(parent)
{

}

QMinMaxHintObject::~QMinMaxHintObject()
{

}

//void QMinMaxHintObject::Triggered(std::shared_ptr<medical_imaging::SceneBase> pScene)
//{
//    emit triggered(pScene);
//}

void QMinMaxHintObject::trigger(const std::string& s)
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

void MouseOpMinMaxHint::press(const QPoint& pt)
{
    
}

void MouseOpMinMaxHint::move(const QPoint& pt)
{

}

void MouseOpMinMaxHint::release(const QPoint& pt)
{

}

void MouseOpMinMaxHint::double_click(const QPoint& pt)
{
    if (m_pMinMaxHintObject)
    {
        m_pMinMaxHintObject->trigger(m_pScene->get_name());
    }
}

void MouseOpMinMaxHint::set_min_max_hint_object(QMinMaxHintObject* pObj)
{
    m_pMinMaxHintObject = pObj;
}

MED_IMAGING_END_NAMESPACE

    