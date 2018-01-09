#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_RESOURCE_MANANGER_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_RESOURCE_MANANGER_H

#include "cudaresource/mi_cuda_resource_export.h"
#include "cudaresource/mi_cuda_resource_define.h"
#include "util/mi_uid.h"
#include <memory>
#include <map>
#include <ostream>
#include <sstream>
#include <boost/thread/mutex.hpp>

MED_IMG_BEGIN_NAMESPACE

class CudaTexture1D;
class CudaTexture2D;
class CudaTexture3D;
class CudaGLTexture2D;
class CudaGlobalMemory;

template<class T>
class CudaResoueceRecord {
public:
    CudaResoueceRecord() {}

    ~CudaResoueceRecord() {}

    const char* get_type_desc() const;
    
    void insert(UIDType uid, std::shared_ptr<T> ptr) {
        boost::mutex::scoped_lock locker(_mutex);
        _records[uid] = ptr;
    }

    float get_memory_used() {
        boost::mutex::scoped_lock locker(_mutex);
        float sum(0.0f);
        for (auto it = _records.begin(); it != _records.end(); ) {
            std::shared_ptr<T> ptr = it->second.lock();
            if (ptr) {
                sum += ptr->memory_used();
                ++it;
            }
            else {
                it = _records.erase(it);
            }
        }
        return sum;
    }

    void clear() {
        boost::mutex::scoped_lock locker(_mutex);
        _records.clear();
    }
    std::string get_specification(const std::string& split = " ") {
        boost::mutex::scoped_lock locker(_mutex);
        std::stringstream ss;
        ss << get_type_desc() << ": [" << split ;
        float sum(0.0f);
        for (auto it = _records.begin(); it != _records.end(); ) {
            std::shared_ptr<T> ptr = it->second.lock();
            if (ptr) {
                sum += ptr->memory_used();
                ss << "{" << ptr->get_description() <<", size: " << ptr->memory_used() << " KB }, " << split;
                ++it;
            }
            else {
                it = _records.erase(it);
            }
        }
        
        ss << "{total size: " << sum << " KB }";
        ss << "]";
        return ss.str();
    }

private:
    std::map<UIDType, std::weak_ptr<T>> _records;
    boost::mutex _mutex;
};

class CUDAResource_Export CudaResourceManager {
public:
    ~CudaResourceManager();

    static CudaResourceManager* instance();
    static void release();
    
    CudaGlobalMemoryPtr create_global_memory(const std::string& desc);
    CudaTexture1DPtr create_cuda_texture_1d(const std::string& desc);
    CudaTexture2DPtr create_cuda_texture_2d(const std::string& desc);
    CudaTexture3DPtr create_cuda_texture_3d(const std::string& desc);
    CudaGLTexture2DPtr create_cuda_gl_texture_2d(const std::string& desc);
    CudaSurface2DPtr create_cuda_surface_2d(const std::string& desc);
    CudaTexture1DArrayPtr create_cuda_texture_1d_array(const std::string& desc, int length);

    float get_device_memory_memory_used();
    float get_cuda_texture_1d_memory_used();
    float get_cuda_texture_2d_memory_used();
    float get_cuda_texture_3d_memory_used();
    float get_cuda_gl_texture_2d_memory_used();
    float get_ccuda_texture_1d_array_memory_used();

    friend std::ostream& operator << (std::ostream& strm, CudaResourceManager& cuda_res_m) {
        strm << cuda_res_m.get_specification(" ");
        return strm;
    }

    std::string get_specification(const std::string& split = " ");

private:
    CudaResourceManager();
    static CudaResourceManager *_instance;
    static boost::mutex _mutex;

    CudaResoueceRecord<CudaGlobalMemory>     _record_global_memory;
    CudaResoueceRecord<CudaTexture1D>        _record_tex_1d;
    CudaResoueceRecord<CudaTexture2D>        _record_tex_2d;
    CudaResoueceRecord<CudaTexture3D>        _record_tex_3d;
    CudaResoueceRecord<CudaGLTexture2D>      _record_gl_tex_2d;
    CudaResoueceRecord<CudaSurface2D>        _record_surface_2d;
    CudaResoueceRecord<CudaTexture1DArray>   _record_texture_1d_array;

private:
    DISALLOW_COPY_AND_ASSIGN(CudaResourceManager);    
};


MED_IMG_END_NAMESPACE
#endif
