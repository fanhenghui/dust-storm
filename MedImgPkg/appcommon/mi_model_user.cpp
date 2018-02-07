#include "mi_model_user.h"

MED_IMG_BEGIN_NAMESPACE

ModelUser::ModelUser() {

}

ModelUser::~ModelUser() {

}   

void ModelUser::set_user_id(const std::string& user_id) {
    _user_id = user_id;
}

const std::string& ModelUser::get_user_id() const {
    return _user_id;
}

void ModelUser::set_user_role(RoleType role) {
    _user_role = role;
}

RoleType ModelUser::get_user_role() const {
    return _user_role;
}

MED_IMG_END_NAMESPACE