#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_SURFACE_2D_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_SURFACE_2D_H

#include "cudaresource/mi_cuda_surface_base.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaSurface2D : public CudaSurfaceBase {
public:
    explicit CudaSurface2D(UIDType uid);
    virtual ~CudaSurface2D();

    virtual float memory_used() const;

    int get_width() const;

    int get_height() const;

    int load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int width, int height, void* h_data);

    int update(int offset_x, int offset_y, int width, int height, void* h_data);

    int download(unsigned int size, void* h_data);
private:
    int _width;
    int _height;
};

MED_IMG_END_NAMESPACE
#endif
