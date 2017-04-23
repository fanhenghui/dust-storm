#include "mi_observer_voi_table.h"

using namespace medical_imaging;

VOITableObserver::VOITableObserver():_nodule_object(nullptr)
{

}

VOITableObserver::~VOITableObserver()
{

}

void VOITableObserver::update()
{
    if (_nodule_object)
    {
        _nodule_object->add_nodule();
    }
}


void VOITableObserver::set_nodule_object(QNoduleObject* obj)
{
    _nodule_object = obj;
}

QNoduleObject::QNoduleObject(QObject* parent /*=0*/)
{

}

void QNoduleObject::delete_nodule(int id)
{
    emit nodule_deleted(id);
}

void QNoduleObject::add_nodule()
{
    emit nodule_added();
}
