#include "mi_model_interface.h"
#include "mi_observer_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

IModel::IModel():m_bIsChanged(false)
{

}

IModel::~IModel()
{

}

void IModel::AddObserver( ObserverPtr pObserver )
{
    bool bIsRegistered = false;
    auto it = m_Observers.begin();
    while(it != m_Observers.end())
    {
        if (*it == pObserver)
        {
            bIsRegistered = true;
            break;
        }
        ++it;
    }
    if (!bIsRegistered)
    {
        m_Observers.push_back(pObserver);
    }
}

void IModel::DeleteObserver( ObserverPtr pObserver )
{
    auto it = m_Observers.begin();
    while(it != m_Observers.end())
    {
        if (*it == pObserver)
        {
            m_Observers.erase(it);
            break;
        }
        ++it;
    }
}

void IModel::NotifyAllObserver()
{
    if (m_bIsChanged)
    {

        for (auto it = m_Observers.begin() ; it != m_Observers.end() ; ++it)
        {
            (*it)->Update();
        }

        m_bIsChanged = false;
    }
}

void IModel::SetChanged()
{
    m_bIsChanged = true;
}

void IModel::CleanChanged()
{
    m_bIsChanged = false;
}

bool IModel::HasChanged()
{
    return m_bIsChanged;
}

MED_IMAGING_END_NAMESPACE