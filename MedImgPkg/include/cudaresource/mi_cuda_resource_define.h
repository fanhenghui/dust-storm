#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_RESOURCE_DEFINE_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_RESOURCE_DEFINE_H

#include <memory>
#include "cudaresource/mi_cuda_resource_export.h"

MED_IMG_BEGIN_NAMESPACE

class CudaTexture1D;
class CudaTexture2D;
class CudaTexture3D;
class CudaGLTexture2D;
class CudaDeviceMemory;

typedef std::shared_ptr<CudaTexture1D>    CudaTexture1DPtr;
typedef std::shared_ptr<CudaTexture2D>    CudaTexture2DPtr;
typedef std::shared_ptr<CudaTexture3D>    CudaTexture3DPtr;
typedef std::shared_ptr<CudaGLTexture2D>  CudaGLTexture2DPtr;
typedef std::shared_ptr<CudaDeviceMemory> CudaDeviceMemoryPtr;

MED_IMG_END_NAMESPACE
#endif