#include "mi_model_interface.h"
#include "mi_observer_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

IModel::IModel():m_bIsChanged(false)
{

}

IModel::~IModel()
{

}

void IModel::add_observer( ObserverPtr pObserver )
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

void IModel::delete_observer( ObserverPtr pObserver )
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

void IModel::notify()
{
    if (m_bIsChanged)
    {

        for (auto it = m_Observers.begin() ; it != m_Observers.end() ; ++it)
        {
            (*it)->update();
        }

        m_bIsChanged = false;
    }
}

void IModel::set_changed()
{
    m_bIsChanged = true;
}

void IModel::reset_changed()
{
    m_bIsChanged = false;
}

bool IModel::has_changed()
{
    return m_bIsChanged;
}

MED_IMAGING_END_NAMESPACE