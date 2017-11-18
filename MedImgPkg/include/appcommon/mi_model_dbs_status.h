#ifndef MED_IMG_APPCOMMON_MI_MODEL_DBS_STATUS_H
#define MED_IMG_APPCOMMON_MI_MODEL_DBS_STATUS_H

#include <string>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelDBSStatus : public IModel {
public:
    ModelDBSStatus() {}
    virtual ~ ModelDBSStatus() {}

    void reset();
    void set_error_info(const std::string& err);
    std::string get_error_info() const;

private:
    std::string _err_info;
private:
    DISALLOW_COPY_AND_ASSIGN(ModelDBSStatus);
};

MED_IMG_END_NAMESPACE
#endif