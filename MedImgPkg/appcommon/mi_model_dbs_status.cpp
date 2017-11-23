#include "mi_model_dbs_status.h"

MED_IMG_BEGIN_NAMESPACE

ModelDBSStatus::ModelDBSStatus():_success(false),_has_preprocess_mask(false),_has_ai_annotation(false) {

}
void ModelDBSStatus::reset() {
    _success = false;
    _has_preprocess_mask = false;
    _has_ai_annotation = false;
    _err_info = "";
}

bool ModelDBSStatus::success() {
    return _success;
}

void ModelDBSStatus::set_success() {
    _success = true;
}

void ModelDBSStatus::set_error_info(const std::string& err) {
    _err_info = err;
}

std::string ModelDBSStatus::get_error_info() const {
    return _err_info;
}

bool ModelDBSStatus::has_preprocess_mask() {
    return _has_preprocess_mask;
}

void ModelDBSStatus::set_preprocess_mask() {
    _has_preprocess_mask = true;
}

bool ModelDBSStatus::has_ai_annotation() {
    return _has_ai_annotation;
}

void ModelDBSStatus::set_ai_annotation() {
    _has_ai_annotation = true;
}

MED_IMG_END_NAMESPACE