#ifndef MEDIMGUTIL_MI_CONFIGURATION_H
#define MEDIMGUTIL_MI_CONFIGURATION_H

#include "util/mi_util_export.h"
#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

enum ProcessingUnitType {
    CPU = 0,
    GPU,
};

class Util_Export Configuration {
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

MED_IMG_END_NAMESPACE
#endif