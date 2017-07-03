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

MED_IMG_BEGIN_NAMESPACE

MouseOpMinMaxHint::MouseOpMinMaxHint():_min_max_hint_object(nullptr)
{

}

MouseOpMinMaxHint::~MouseOpMinMaxHint()
{

}

void MouseOpMinMaxHint::press(const QPointF& pt)
{
    //std::cout << "single click \n";
}

void MouseOpMinMaxHint::move(const QPointF& pt)
{

}

void MouseOpMinMaxHint::release(const QPointF& pt)
{

}

void MouseOpMinMaxHint::double_click(const QPointF& pt)
{
    //std::cout << "double click \n";
    if (_min_max_hint_object)
    {
        _min_max_hint_object->trigger(_scene->get_name());
    }
}

void MouseOpMinMaxHint::set_min_max_hint_object(QMinMaxHintObject* obj)
{
    _min_max_hint_object = obj;
}

void MouseOpMinMaxHint::wheel_slide(int)
{

}

MED_IMG_END_NAMESPACE

    