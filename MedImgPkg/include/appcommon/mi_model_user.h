#ifndef MED_IMG_APPCOMMON_MI_MODEL_USER_H
#define MED_IMG_APPCOMMON_MI_MODEL_USER_H

#include <string>
#include <vector>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"
#include "io/mi_dicom_info.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelUser : public IModel {
public:
    ModelUser();
    ~ModelUser();
    
    void set_user_id(const std::string& user_id);
    const std::string& get_user_id() const;

    void set_user_role(RoleType role);
    RoleType get_user_role() const;

private:
    std::string _user_id;
    RoleType _user_role;

private:
    DISALLOW_COPY_AND_ASSIGN(ModelUser);
};

MED_IMG_END_NAMESPACE

#endif