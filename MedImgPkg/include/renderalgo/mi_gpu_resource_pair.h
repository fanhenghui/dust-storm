#ifndef MED_IMG_RENDERALGORITHM_MI_GPU_RESOURCE_PAIR_H
#define MED_IMG_RENDERALGORITHM_MI_GPU_RESOURCE_PAIR_H

#include <memory>
#include "renderalgo/mi_render_algo_export.h"
#include "glresource/mi_gl_resource_define.h"
#include "cudaresource/mi_cuda_resource_define.h"

MED_IMG_BEGIN_NAMESPACE

template<class GLType , class CudaType>
class GPUResourcePair
{
public:
    explicit GPUResourcePair(std::shared_ptr<GLType> res):_gl_res(res) {}

    explicit GPUResourcePair(std::shared_ptr<CudaType> res) :_cuda_res(res) {}

    inline std::shared_ptr<GLType> get_gl_resource() const {
        return _gl_res;
    }

    inline std::shared_ptr<CudaType> get_cuda_resource() const {
        return _cuda_res;
    }

    inline bool gl() const {
        return _gl_res != nullptr;
    }
    
    inline bool cuda() const {
        return _cuda_res != nullptr;
    }

private:
    std::shared_ptr<GLType> _gl_res;
    std::shared_ptr<CudaType> _cuda_res;
};


typedef GPUResourcePair<GLTexture1D, CudaTexture1D> GPUTexture1DPair;
typedef GPUResourcePair<GLTexture2D, CudaTexture2D> GPUTexture2DPair;
typedef GPUResourcePair<GLTexture3D, CudaTexture3D> GPUTexture3DPair;
typedef GPUResourcePair<GLBuffer, CudaGlobalMemory> GPUMemoryPair;
typedef GPUResourcePair<GLTexture1DArray, CudaTexture1DArray> GPUTexture1DArrayPair;
typedef GPUResourcePair<GLTexture2D, CudaSurface2D> GPUCanvasPair;

typedef std::shared_ptr<GPUTexture1DPair> GPUTexture1DPairPtr;
typedef std::shared_ptr<GPUTexture2DPair> GPUTexture2DPairPtr;
typedef std::shared_ptr<GPUTexture3DPair> GPUTexture3DPairPtr;
typedef std::shared_ptr<GPUMemoryPair> GPUMemoryPairPtr;
typedef std::shared_ptr<GPUTexture1DArrayPair> GPUTexture1DArrayPairPtr;
typedef std::shared_ptr<GPUCanvasPair> GPUCanvasPairPtr;

MED_IMG_END_NAMESPACE

#endif
