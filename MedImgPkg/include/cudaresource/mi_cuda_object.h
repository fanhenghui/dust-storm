#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_OBJECT_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_OBJECT_H

#include <string>
#include "cudaresource/mi_cuda_resource_export.h"
#include "util/mi_uid.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaObject {
public:
    explicit CudaObject(UIDType uid, const std::string& type) : m_uid(uid), m_type(type) {}
    virtual ~CudaObject() {}

    UIDType get_uid() const {
        return m_uid;
    }

    void set_type(const std::string& type) {
        m_type = type;
    }

    std::string get_type() const {
        return m_type;
    }

    std::string get_description() const {
        return _description;
    }

    void set_description(const std::string& des) {
        _description = des;
    }

    friend std::ostream& operator << (std::ostream& strm, const CudaObject& obj) {
        strm << "GLOBJ type: " << obj.get_type() << ", uid: " << obj.get_uid() << ", des: " << obj.get_description();
        return strm;
    }

    friend std::ostream& operator << (std::ostream& strm, const std::shared_ptr<CudaObject>& obj) {
        strm << "GLOBJ type: " << obj->get_type() << ", uid: " << obj->get_uid() << ", des: " << obj->get_description();
        return strm;
    }

    virtual void initialize() = 0;
    virtual void finalize() = 0;

private:
    UIDType m_uid;
    std::string m_type;
    std::string _description;
};

MED_IMG_END_NAMESPACE
#endif
