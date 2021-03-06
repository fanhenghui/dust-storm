#ifndef MEDIMGRENDERALGO_VOLUME_INFOS_H
#define MEDIMGRENDERALGO_VOLUME_INFOS_H

#include "renderalgo/mi_render_algo_export.h"

#include "arithmetic/mi_aabb.h"

#include "renderalgo/mi_gpu_resource_pair.h"
#include "renderalgo/mi_brick_define.h"
#include "renderalgo/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE

class ImageData;
class ImageDataHeader;
class BrickPool;
class CameraCalculator;

// Volume info for upload/update volume&mask(brick & brick info ) to host&device
class RenderAlgo_Export VolumeInfos {
public:
    explicit VolumeInfos(RayCastingStrategy strategy,  GPUPlatform p);
    ~VolumeInfos();

    void refresh();

    void finialize();

    void set_volume(std::shared_ptr<ImageData> image_data);
    void set_mask(std::shared_ptr<ImageData> image_data);
    void set_data_header(std::shared_ptr<ImageDataHeader> data_header);

    void cache_original_mask();//cache current mask to original
    std::shared_ptr<ImageData> get_cache_original_mask();

    std::shared_ptr<CameraCalculator> get_camera_calculator();

    GPUTexture3DPairPtr get_volume_texture();
    GPUTexture3DPairPtr get_mask_texture();

    std::shared_ptr<ImageData> get_volume();
    std::shared_ptr<ImageData> get_mask();
    std::shared_ptr<ImageDataHeader> get_data_header();
    std::shared_ptr<BrickPool> get_brick_pool();

    // update(should update to host and device)
    void update_mask(const unsigned int (&begin)[3], const unsigned int (&end)[3],
                     unsigned char* data_updated,
                     bool has_data_array_changed = true);

private:
    void refresh_upload_volume();
    void refresh_upload_mask();
    void refresh_update_mask();
    void refresh_stored_mask_brick_info();
    void refresh_cache_mask_brick_info();

private:
    RayCastingStrategy _strategy;
    GPUPlatform _gpu_platform;

    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<ImageData> _mask_data;
    std::shared_ptr<ImageDataHeader> _data_header;

    std::shared_ptr<ImageData> _cache_original_mask;//cache for recover

    // P.S here will use vector for separate volume later
    GPUTexture3DPairPtr _volume_texture; 
    GPUTexture3DPairPtr _mask_texture;

    std::shared_ptr<BrickPool> _brick_pool;

    std::shared_ptr<CameraCalculator> _camera_calculator;

    // cache
    // volume upload
    bool _volume_dirty;
    // mask upload
    bool _mask_dirty;
    // mask update
    std::vector<AABBUI> _mask_aabb_to_be_update;
    std::vector<unsigned char*> _mask_array_to_be_update;

private:
    DISALLOW_COPY_AND_ASSIGN(VolumeInfos);
};

MED_IMG_END_NAMESPACE

#endif