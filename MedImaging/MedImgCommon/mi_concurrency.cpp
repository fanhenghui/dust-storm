#include "mi_concurrency.h"
#include "boost/thread.hpp"

MED_IMAGING_BEGIN_NAMESPACE

Concurrency* Concurrency::m_instance = nullptr;

boost::mutex Concurrency::m_mutexStatic;

Concurrency* Concurrency::instance()
{
    if (!m_instance)
    {
        boost::unique_lock<boost::mutex> locker(m_mutexStatic);
        if (!m_instance)
        {
            m_instance = new Concurrency();
        }
    }
    return m_instance;
}

Concurrency::~Concurrency()
{

}

Concurrency::Concurrency()
{
    m_uiHardwareConcurrency = boost::thread::hardware_concurrency();
    m_uiAppConcurrency = m_uiHardwareConcurrency;
}

void Concurrency::set_app_concurrency(unsigned int uiValue)
{
    m_uiAppConcurrency = uiValue > m_uiHardwareConcurrency ? m_uiHardwareConcurrency : uiValue;
}

unsigned int Concurrency::get_app_concurrency()
{
    return m_uiAppConcurrency;
}

unsigned int Concurrency::get_hardware_concurrency()
{
    return m_uiHardwareConcurrency;
}

MED_IMAGING_END_NAMESPACE