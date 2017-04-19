#ifndef MED_IMAGING_CONFIGURATION_H
#define MED_IMAGING_CONFIGURATION_H

#include "MedImgCommon/mi_common_stdafx.h"
#include "MedImgCommon/mi_common_define.h"

#include "boost/thread/mutex.hpp"

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export Configuration
{
public:
    static Configuration* Instance();
    ~Configuration();
    ProcessingUnitType GetProcessingUnitType();
    void SetProcessingUnitType(ProcessingUnitType eType);//For testing
protected:
private:
    Configuration();

    static Configuration* m_instance;
    static boost::mutex m_mutex;

private:
    ProcessingUnitType m_ePUT;
};

MED_IMAGING_END_NAMESPACE
#endif