#include "mi_observer_voi_table.h"

using namespace MedImaging;
const std::string ksNoduleTypeGGN = std::string("GGN");
const std::string ksNoduleTypeAAH = std::string("AAH");

VOITableObserver::VOITableObserver():m_pNoduleObject(nullptr)
{

}

VOITableObserver::~VOITableObserver()
{

}

void VOITableObserver::Update()
{
    if (m_pNoduleObject)
    {
        m_pNoduleObject->AddNodule();
    }
}


void VOITableObserver::SetNoduleObject(QNoduleObject* pAddNodule)
{
    m_pNoduleObject = pAddNodule;
}

QNoduleObject::QNoduleObject(QObject* parent /*=0*/)
{

}

void QNoduleObject::DeleteNodule(int id)
{
    emit deleteNodule(id);
}

void QNoduleObject::AddNodule()
{
    emit addNodule();
}
