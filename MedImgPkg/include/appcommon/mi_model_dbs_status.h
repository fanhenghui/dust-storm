#ifndef MED_IMG_APPCOMMON_MI_MODEL_DBS_STATUS_H
#define MED_IMG_APPCOMMON_MI_MODEL_DBS_STATUS_H

#include <string>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelDBSStatus : public IModel {
public:
    ModelDBSStatus();
    virtual ~ ModelDBSStatus() {}

    void reset();

    //success tag
    bool success();
    void set_success();

    //error msg
    void set_error_info(const std::string& err);
    std::string get_error_info() const;

    //preprocess mask query 
    bool has_preprocess_mask();
    void set_preprocess_mask();

private:
    bool _success;
    std::string _err_info;
    bool _has_preprocess_mask;
private:
    DISALLOW_COPY_AND_ASSIGN(ModelDBSStatus);
};

MED_IMG_END_NAMESPACE
#endif