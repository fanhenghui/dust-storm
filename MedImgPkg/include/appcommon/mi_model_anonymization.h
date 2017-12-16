#ifndef MED_IMG_APPCOMMON_MI_MODEL_ANONYMIZATION_H
#define MED_IMG_APPCOMMON_MI_MODEL_ANONYMIZATION_H

#include <string>
#include <vector>
#include <boost/thread/mutex.hpp>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelAnonymization : public IModel { 
public:
    ModelAnonymization();
    ~ModelAnonymization();

    void set_anonymization_flag(bool flag);
    bool get_anonymization_flag() const;

    //TODO configure DCM tag ?

private:
    bool _annoymization_flag;
    mutable boost::mutex _mutex;
private:
    DISALLOW_COPY_AND_ASSIGN(ModelAnonymization);
};

MED_IMG_END_NAMESPACE
#endif