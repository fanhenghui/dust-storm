#include "mi_cuda_resource_manager.h"

#include "util/mi_uid.h"

#include "mi_cuda_texture_1d.h"
#include "mi_cuda_texture_2d.h"
#include "mi_cuda_texture_3d.h"
#include "mi_cuda_gl_texture_2d.h"
#include "mi_cuda_global_memory.h"
#include "mi_cuda_surface_2d.h"
#include "mi_cuda_texture_1d_array.h"
#include "mi_cuda_time_query.h"

MED_IMG_BEGIN_NAMESPACE

CudaResourceManager* CudaResourceManager::_instance = nullptr;
boost::mutex CudaResourceManager::_mutex;

CudaResourceManager::CudaResourceManager():_uid_generator(new UIDGenerator()) {
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
            delete[] _instance;
            _instance = nullptr;
        }
    }
}

CudaGlobalMemoryPtr CudaResourceManager::create_global_memory(const std::string& desc) {
    UIDType uid = _uid_generator->tick();
    CudaGlobalMemoryPtr ptr(new CudaGlobalMemory(uid));
    ptr->set_description(desc);
    _record_global_memory.insert(uid, ptr);
    return ptr;
}

CudaTexture1DPtr CudaResourceManager::create_cuda_texture_1d(const std::string& desc) {
    UIDType uid = _uid_generator->tick();
    CudaTexture1DPtr ptr(new CudaTexture1D(uid));
    ptr->set_description(desc);
    _record_tex_1d.insert(uid, ptr);
    return ptr;
}

CudaTexture2DPtr CudaResourceManager::create_cuda_texture_2d(const std::string& desc) {
    UIDType uid = _uid_generator->tick();
    CudaTexture2DPtr  ptr(new CudaTexture2D(uid));
    ptr->set_description(desc);
    _record_tex_2d.insert(uid, ptr);
    return ptr;
}

CudaTexture3DPtr CudaResourceManager::create_cuda_texture_3d(const std::string& desc) {
    UIDType uid = _uid_generator->tick();
    CudaTexture3DPtr ptr(new CudaTexture3D(uid));
    ptr->set_description(desc);
    _record_tex_3d.insert(uid, ptr);
    return ptr;
}

CudaGLTexture2DPtr CudaResourceManager::create_cuda_gl_texture_2d(const std::string& desc) {
    UIDType uid = _uid_generator->tick();
    CudaGLTexture2DPtr ptr(new CudaGLTexture2D(uid));
    ptr->set_description(desc);
    _record_gl_tex_2d.insert(uid, ptr);
    return ptr;
}

CudaSurface2DPtr CudaResourceManager::create_cuda_surface_2d(const std::string& desc) {
    UIDType uid = _uid_generator->tick();
    CudaSurface2DPtr ptr(new CudaSurface2D(uid));
    ptr->set_description(desc);
    _record_surface_2d.insert(uid, ptr);
    return ptr;
}

CudaTexture1DArrayPtr CudaResourceManager::create_cuda_texture_1d_array(const std::string& desc, int length) {
    UIDType uid = _uid_generator->tick();
    CudaTexture1DArrayPtr ptr(new CudaTexture1DArray(uid, length));
    ptr->set_description(desc);
    _record_texture_1d_array.insert(uid, ptr);
    return ptr;
}

CudaTimeQueryPtr CudaResourceManager::create_cuda_time_query(const std::string& desc) {
    UIDType uid = _uid_generator->tick();
    CudaTimeQueryPtr ptr(new CudaTimeQuery(uid));
    ptr->set_description(desc);
    _record_time_query.insert(uid, ptr);
    return ptr;
}

float CudaResourceManager::get_device_memory_memory_used() {
    return _record_global_memory.get_memory_used();
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

float CudaResourceManager::get_cuda_texture_1d_array_memory_used() {
    return _record_texture_1d_array.get_memory_used();
}

float CudaResourceManager::get_time_query_memory_used() {
    return _record_time_query.get_memory_used();
}

std::string CudaResourceManager::get_specification(const std::string& split) {
    std::stringstream ss;
    ss << "CUDA Resources: [" << split <<
        _record_global_memory.get_specification(split) << ", " << split <<
        _record_tex_1d.get_specification(split) << ", " << split <<
        _record_tex_2d.get_specification(split) << ", " << split <<
        _record_tex_3d.get_specification(split) << ", " << split <<
        _record_gl_tex_2d.get_specification(split) << ", " << split << "]";
    return ss.str();
}

template<>
const char* CudaResoueceRecord<CudaGlobalMemory>::get_type_desc() const {
    return "CudaGlobalMemory";
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

template<>
const char* CudaResoueceRecord<CudaTexture1DArray>::get_type_desc() const {
    return "CudaTexture1DArray";
}

template<>
const char* CudaResoueceRecord<CudaSurface2D>::get_type_desc() const {
    return "CudaSurface2D";
}

template<>
const char* CudaResoueceRecord<CudaTimeQuery>::get_type_desc() const {
    return "CudaTimeQuery";
}


MED_IMG_END_NAMESPACE