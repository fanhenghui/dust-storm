#ifndef MEDIMGRENDERALGO_BRICK_INFO_CALCULATOR_H
#define MEDIMGRENDERALGO_BRICK_INFO_CALCULATOR_H

#include "renderalgo/mi_render_algo_export.h"

#include "arithmetic/mi_aabb.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_brick_define.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_gpu_resource_pair.h"

#include <memory>

MED_IMG_BEGIN_NAMESPACE

class ImageData;
class BrickPool;

class RenderAlgo_Export BrickInfoCalculator {
public:
    explicit BrickInfoCalculator(GPUPlatform p);
    virtual ~BrickInfoCalculator();

    void set_data(std::shared_ptr<ImageData> img);
    void set_data_texture(GPUTexture3DPairPtr tex);
    void set_brick_info_buffer(GPUMemoryPairPtr info_buffer);
    void set_brick_info_array(char* info_array);

    void set_brick_size(unsigned int brick_size);
    void set_brick_dim(unsigned int (&brick_dim)[3]);
    void set_brick_margin(unsigned int brick_margin);

    virtual void calculate() = 0;

protected:
    GPUPlatform _gpu_platform;
    std::shared_ptr<ImageData> _img_data;
    GPUTexture3DPairPtr _img_texture;
    GPUMemoryPairPtr _info_buffer;
    char* _info_array;

    unsigned int _brick_size;
    unsigned int _brick_dim[3];
    unsigned int _brick_margin;

    GLResourceShield _res_shield;


private:
    DISALLOW_COPY_AND_ASSIGN(BrickInfoCalculator);
};

class RenderAlgo_Export VolumeBrickInfoCalculator : public BrickInfoCalculator {
public:
    explicit VolumeBrickInfoCalculator(GPUPlatform p);
    virtual ~VolumeBrickInfoCalculator();

    virtual void calculate();

private:
    void initialize();

    void calculate_gl();
    void download_gl();

    void calculate_cuda();
    void download_cuda();

private:
    GLProgramPtr _gl_program;
};

class RenderAlgo_Export MaskBrickInfoCalculator : public BrickInfoCalculator {
public:
    explicit MaskBrickInfoCalculator(GPUPlatform p);
    virtual ~MaskBrickInfoCalculator();

    void set_visible_labels(const std::vector<unsigned char>& labels);

    virtual void calculate();
    void update(const AABBUI& aabb);

private:
    void initialize();

    void update_gl(const AABBUI& aabb);
    void download_gl();

    void update_cuda(const AABBUI& aabb);
    void download_cuda();

private:
    GLProgramPtr _gl_program;
    GLBufferPtr _gl_buffer_visible_labels;
    CudaDeviceMemoryPtr _cuda_memory;
    std::vector<unsigned char> _visible_labels;
};

MED_IMG_END_NAMESPACE

#endif