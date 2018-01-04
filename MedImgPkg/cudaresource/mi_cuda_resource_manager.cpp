#include "mi_cuda_resource_manager.h"

#include "mi_cuda_texture_1d.h"
#include "mi_cuda_texture_2d.h"
#include "mi_cuda_texture_3d.h"
#include "mi_cuda_gl_texture_2d.h"
#include "mi_cuda_device_memory.h"

MED_IMG_BEGIN_NAMESPACE

CudaResourceManager* CudaResourceManager::_instance = nullptr;
boost::mutex CudaResourceManager::_mutex;

CudaResourceManager::CudaResourceManager() {
}

CudaResourceManager::~CudaResourceManager() {
}

CudaResourceManager* CudaResourceManager::instance() {
    if (nullptr == _instance) {
        boost::mutex::scoped_lock locker(_mutex);
        if (nullptr == _instance) {
            _instance = new CudaResourceManager();
        }
    }
    return _instance;
}

void CudaResourceManager::release() {
    if (_instance) {
        boost::mutex::scoped_lock locker(_mutex);
        if (nullptr == _instance) {
            delete [] _instance;
            _instance = nullptr;
        }
    }
}

std::shared_ptr<CudaDeviceMemory> CudaResourceManager::create_device_memory(const std::string& desc) {
    UIDType uid = 0;
    std::shared_ptr<CudaDeviceMemory> ptr(new CudaDeviceMemory(uid));
    ptr->set_description(desc);
    _record_device_memory.insert(uid, ptr);
    return ptr;
}

std::shared_ptr<CudaTexture1D> CudaResourceManager::create_cuda_texture_1d(const std::string& desc) {
    UIDType uid = 0;
    std::shared_ptr<CudaTexture1D> ptr(new CudaTexture1D(uid));
    ptr->set_description(desc);
    _record_tex_1d.insert(uid, ptr);
    return ptr;
}

std::shared_ptr<CudaTexture2D> CudaResourceManager::create_cuda_texture_2d(const std::string& desc) {
    UIDType uid = 0;
    std::shared_ptr<CudaTexture2D> ptr(new CudaTexture2D(uid));
    ptr->set_description(desc);
    _record_tex_2d.insert(uid, ptr);
    return ptr;
}

std::shared_ptr<CudaTexture3D> CudaResourceManager::create_cuda_texture_3d(const std::string& desc) {
    UIDType uid = 0;
    std::shared_ptr<CudaTexture3D> ptr(new CudaTexture3D(uid));
    ptr->set_description(desc);
    _record_tex_3d.insert(uid, ptr);
    return ptr;
}

std::shared_ptr<CudaGLTexture2D> CudaResourceManager::create_cuda_gl_texture_2d(const std::string& desc) {
    UIDType uid = 0;
    std::shared_ptr<CudaGLTexture2D> ptr(new CudaGLTexture2D(uid));
    ptr->set_description(desc);
    _record_gl_tex_2d.insert(uid, ptr);
    return ptr;
}

float CudaResourceManager::get_device_memory_memory_used() {
    return _record_device_memory.get_memory_used();
}

float CudaResourceManager::get_cuda_texture_1d_memory_used() {
    return _record_tex_1d.get_memory_used();
}

float CudaResourceManager::get_cuda_texture_2d_memory_used() {
    return _record_tex_2d.get_memory_used();
}

float CudaResourceManager::get_cuda_texture_3d_memory_used() {
    return _record_tex_3d.get_memory_used();
}

float CudaResourceManager::get_cuda_gl_texture_2d_memory_used() {
    return _record_gl_tex_2d.get_memory_used();
}

std::string CudaResourceManager::get_specification(const std::string& split) {
    std::stringstream ss;
    ss << "CUDA Resources: [" << split << 
        _record_device_memory.get_specification(split) << ", " << split <<
        _record_tex_1d.get_specification(split) << ", " << split <<
        _record_tex_2d.get_specification(split) << ", " << split <<
        _record_tex_3d.get_specification(split) << ", " << split <<
        _record_gl_tex_2d.get_specification(split) << ", " << split << "]";
    return ss.str();
}

template<>
const char* CudaResoueceRecord<CudaDeviceMemory>::get_type_desc() const {
    return "CudaDeviceMemory";
}

template<>
const char* CudaResoueceRecord<CudaTexture1D>::get_type_desc() const {
    return "CudaTexture1D";
}

template<>
const char* CudaResoueceRecord<CudaTexture2D>::get_type_desc() const {
    return "CudaTexture2D";
}

template<>
const char* CudaResoueceRecord<CudaTexture3D>::get_type_desc() const {
    return "CudaTexture3D";
}

template<>
const char* CudaResoueceRecord<CudaGLTexture2D>::get_type_desc() const {
    return "CudaGLTexture2D";
}

MED_IMG_END_NAMESPACE