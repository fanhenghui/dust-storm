#include "mi_model_dbs_status.h"

MED_IMG_BEGIN_NAMESPACE

void ModelDBSStatus::reset() {
    _err_info = "";
}

void ModelDBSStatus::set_error_info(const std::string& err) {
    _err_info = err;
}

std::string ModelDBSStatus::get_error_info() const {
    return _err_info;
}

MED_IMG_END_NAMESPACE