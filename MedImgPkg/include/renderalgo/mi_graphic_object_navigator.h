#ifndef MEDIMGRENDERALGO_MI_GRIPHIC_OBJECT_NAVIGATOR_H
#define MEDIMGRENDERALGO_MI_GRIPHIC_OBJECT_NAVIGATOR_H

#include "renderalgo/mi_graphic_object_interface.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_gpu_resource_pair.h"

MED_IMG_BEGIN_NAMESPACE

class GraphicObjectNavigator : public IGraphicObject {
public:
    explicit GraphicObjectNavigator(GPUPlatform platform);
    virtual ~GraphicObjectNavigator();

    virtual void initialize();
    virtual void render(int code = 0);
    void render_to_cuda_surface(CudaSurface2DPtr surface);
    void set_navi_position(int x, int y, int width, int height);

private:
    GPUPlatform _gpu_platform;

    GPUTexture2DPairPtr _navi_tex;
    GLResourceShield _res_shield;
    bool _has_init;

    int _x;
    int _y;
    int _width;
    int _height;

    struct InnerCudaResource;
    std::unique_ptr<InnerCudaResource> _inner_cuda_res;
};

MED_IMG_END_NAMESPACE
#endif