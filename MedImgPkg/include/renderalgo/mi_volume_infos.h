#ifndef MED_IMG_VOLUME_INFOS_H
#define MED_IMG_VOLUME_INFOS_H

#include "renderalgo/mi_render_algo_export.h"
#include "glresource/mi_gl_resource_define.h"
#include "renderalgo/mi_brick_define.h"
#include "arithmetic/mi_aabb.h"

MED_IMG_BEGIN_NAMESPACE

class ImageData;
class ImageDataHeader;
class BrickPool;
class CameraCalculator;

// Volume info for upload/update volume&mask(brick & brick info ) to CPU&GPU
class RenderAlgo_Export VolumeInfos
{
public:
    VolumeInfos();
    ~VolumeInfos();

    void refresh();

    void finialize();

    void set_volume(std::shared_ptr<ImageData> image_data);
    void set_mask(std::shared_ptr<ImageData> image_data);
    void set_data_header(std::shared_ptr<ImageDataHeader> data_header);

    std::shared_ptr<CameraCalculator> get_camera_calculator();

    std::vector<GLTexture3DPtr> get_volume_texture();
    std::vector<GLTexture3DPtr> get_mask_texture();

    std::shared_ptr<ImageData> get_volume();
    std::shared_ptr<ImageData> get_mask();
    std::shared_ptr<ImageDataHeader> get_data_header();
    std::shared_ptr<BrickPool> get_brick_pool();

    //update(should update to CPU and GPU)
    void update_mask(const unsigned int (&begin)[3] ,const unsigned int (&end)[3] , unsigned char* data_updated , bool has_data_array_changed = true);

private:
    void refresh_upload_volume_i();
    void refresh_upload_mask_i();
    void refresh_update_mask_i();

private:
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<ImageData> _mask_data;
    std::shared_ptr<ImageDataHeader> _data_header;

    std::vector<GLTexture3DPtr> _volume_textures;//P.S here use vector for separate volume later
    std::vector<GLTexture3DPtr> _mask_textures;

    std::shared_ptr<BrickPool> _brick_pool;

    std::shared_ptr<CameraCalculator> _camera_calculator;

    //cache
    //volume upload
    bool _volume_dirty;
    //mask upload
    bool _mask_dirty;
    //mask update
    std::vector<AABBUI> _mask_aabb_to_be_update;
    std::vector<unsigned char*> _mask_array_to_be_update;

};

MED_IMG_END_NAMESPACE

#endif