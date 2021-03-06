#include "mi_model_dbs_status.h"

MED_IMG_BEGIN_NAMESPACE

ModelDBSStatus::ModelDBSStatus():_success(false),_has_preprocess_mask(false),_has_ai_annotation(false),
_query_ai_annotation(false),_init(false) {

}

void ModelDBSStatus::reset() {
    _success = false;
    _has_preprocess_mask = false;
    _has_ai_annotation = false;
    _query_ai_annotation = false;
    _init = false;
    _err_infos.clear();
}

bool ModelDBSStatus::success() {
    return _success;
}

void ModelDBSStatus::set_success() {
    _success = true;
}

void ModelDBSStatus::push_error_info(const std::string& err) {
    _err_infos.push_back(err);
}

std::vector<std::string> ModelDBSStatus::get_error_infos() const {
    return _err_infos;
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

void ModelDBSStatus::await() {
    boost::mutex::scoped_lock locker(_mutex);
    _condition.wait(_mutex);
}

void ModelDBSStatus::unlock() {
    _condition.notify_one();
}

bool ModelDBSStatus::has_query_ai_annotation() {
    return _query_ai_annotation;
}

void ModelDBSStatus::cancel_ai_annotation() {
    _query_ai_annotation = false;
}

void ModelDBSStatus::query_ai_annotation() {
    _query_ai_annotation = true;
}

bool ModelDBSStatus::has_init() {
    return _init;
}

void ModelDBSStatus::set_init() {
    _init = true;
}

MED_IMG_END_NAMESPACE