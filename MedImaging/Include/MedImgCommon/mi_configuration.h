#ifndef MED_IMAGING_CONFIGURATION_H
#define MED_IMAGING_CONFIGURATION_H

#include "MedImgCommon/mi_common_stdafx.h"
#include "MedImgCommon/mi_common_define.h"

#include "boost/thread/mutex.hpp"

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export Configuration
{
public:
    static Configuration* instance();
    ~Configuration();
    ProcessingUnitType get_processing_unit_type();
    void set_processing_unit_type(ProcessingUnitType type);//For testing
protected:
private:
    Configuration();

    static Configuration* _s_instance;
    static boost::mutex _s_mutex;

private:
    ProcessingUnitType _processing_unit_type;
};

MED_IMAGING_END_NAMESPACE
#endif