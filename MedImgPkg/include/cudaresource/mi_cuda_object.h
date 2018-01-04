#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_OBJECT_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_OBJECT_H

#include <string>
#include "cudaresource/mi_cuda_resource_export.h"
#include "util/mi_uid.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaObject {
public:
    CudaObject(UIDType uid, const std::string& type);
    virtual ~CudaObject();

    virtual void initialize() = 0;
    virtual void finalize() = 0;
    virtual float memory_used() const = 0;//KB

    UIDType get_uid() const;

    void set_type(const std::string& type);
    std::string get_type() const;

    std::string get_description() const;
    void set_description(const std::string& des);

    friend std::ostream& operator << (std::ostream& strm, const CudaObject& obj);
    friend std::ostream& operator << (std::ostream& strm, const std::shared_ptr<CudaObject>& obj);

private:
    struct InnerParams;
    std::unique_ptr<InnerParams> _inner_param;
};

inline std::ostream& operator << (std::ostream& strm, const CudaObject& obj) {
    strm << "CUDAOBJ type: " << obj.get_type() << ", uid: " << obj.get_uid() << ", des: " << obj.get_description();
    return strm;
}

inline std::ostream& operator << (std::ostream& strm, const std::shared_ptr<CudaObject>& obj) {
    strm << "CUDAOBJ type: " << obj->get_type() << ", uid: " << obj->get_uid() << ", des: " << obj->get_description();
    return strm;
}


MED_IMG_END_NAMESPACE
#endif
