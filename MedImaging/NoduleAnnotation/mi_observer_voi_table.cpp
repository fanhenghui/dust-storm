#include "mi_observer_voi_table.h"

using namespace medical_imaging;
const std::string ksNoduleTypeGGN = std::string("GGN");
const std::string ksNoduleTypeAAH = std::string("AAH");

VOITableObserver::VOITableObserver():m_pNoduleObject(nullptr)
{

}

VOITableObserver::~VOITableObserver()
{

}

void VOITableObserver::update()
{
    if (m_pNoduleObject)
    {
        m_pNoduleObject->add_nodule();
    }
}


void VOITableObserver::set_nodule_object(QNoduleObject* pAddNodule)
{
    m_pNoduleObject = pAddNodule;
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
