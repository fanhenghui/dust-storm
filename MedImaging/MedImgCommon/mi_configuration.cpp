#include "mi_configuration.h"

MED_IMAGING_BEGIN_NAMESPACE

boost::mutex Configuration::m_mutex;

Configuration* Configuration::m_instance = nullptr;

Configuration* Configuration::Instance()
{
    if (!m_instance)
    {
        boost::unique_lock<boost::mutex> locker(m_mutex);
        if (!m_instance)
        {
            m_instance= new Configuration();
        }
    }
    return m_instance;
}

Configuration::~Configuration()
{

}

ProcessingUnitType Configuration::GetProcessingUnitType()
{
    return m_ePUT;
}

Configuration::Configuration():m_ePUT(CPU)
{
    //Check hardware processing unit . Check if has GPU
}

void Configuration::SetProcessingUnitType(ProcessingUnitType eType)
{
    m_ePUT = eType;
}

MED_IMAGING_END_NAMESPACE