#ifndef MED_IMG_CONFIGURATION_H
#define MED_IMG_CONFIGURATION_H

#include "MedImgUtil/mi_util_export.h"
#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

enum ProcessingUnitType
{
    CPU = 0,
    GPU,
};

class Util_Export Configuration
{
public:
    static Configuration* instance();

    ~Configuration();

    ProcessingUnitType get_processing_unit_type();
    void set_processing_unit_type(ProcessingUnitType type);//For testing

    void set_nodule_file_rsa(bool b);
    bool get_nodule_file_rsa();

protected:
private:
    Configuration();

    static Configuration* _s_instance;
    static boost::mutex _s_mutex;

private:
    ProcessingUnitType _processing_unit_type;
    bool _is_nodule_file_rsa;
};

MED_IMG_END_NAMESPACE
#endif