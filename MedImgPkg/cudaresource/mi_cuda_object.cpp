#include "mi_cuda_object.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

struct CudaObject::InnerParams {
    UIDType uid;
    std::string type;
    std::string description;
    InnerParams(UIDType uid_, const std::string& type_):uid(uid_), type(type_) {}
};

CudaObject::CudaObject(UIDType uid, const std::string& type) : _inner_param(new InnerParams(uid, type)) {}

CudaObject::~CudaObject() {
    MI_CUDARESOURCE_LOG(MI_INFO) << "{type: " << this->get_type() << " uid: " << this->get_uid() << ", desc: " << this->get_description() << "} release.";
}

UIDType CudaObject::get_uid() const {
    return _inner_param->uid;
}

void CudaObject::set_type(const std::string& type) {
    _inner_param->type = type;
}

std::string CudaObject::get_type() const {
    return _inner_param->type;
}

std::string CudaObject::get_description() const {
    return _inner_param->description;
}

void CudaObject::set_description(const std::string& des) {
    _inner_param->description = des;
}



MED_IMG_END_NAMESPACE