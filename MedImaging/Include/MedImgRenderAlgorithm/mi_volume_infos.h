#ifndef MED_IMAGING_VOLUME_INFOS_H
#define MED_IMAGING_VOLUME_INFOS_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

MED_IMAGING_BEGIN_NAMESPACE

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

    void finialize();

    //Set input
    void set_volume(std::shared_ptr<ImageData> image_data);
    void set_mask(std::shared_ptr<ImageData> image_data);
    void set_data_header(std::shared_ptr<ImageDataHeader> data_header);

    //Get camera calculator
    std::shared_ptr<CameraCalculator> get_camera_calculator();

    //Get GPU resource
    std::vector<GLTexture3DPtr> get_volume_texture();
    std::vector<GLTexture3DPtr> get_mask_texture();

    GLBufferPtr get_volume_brick_info_buffer();
    //GLBufferPtr GetMaskBrickInfoBuffer(const std::vector<unsigned char>& vis_labels);

    //Get GPU resource
    std::shared_ptr<ImageData> get_volume();
    std::shared_ptr<ImageData> get_mask();
    std::shared_ptr<ImageDataHeader> get_data_header();

    BrickCorner* get_brick_corner();
    BrickUnit* get_volume_brick_unit();
    BrickUnit* get_mask_brick_unit();
    VolumeBrickInfo* get_volume_brick_info();
    MaskBrickInfo* get_mask_brick_info(const std::vector<unsigned char>& vis_labels);

    //update(should update to CPU and GPU)
    void update_volume(unsigned int (&begin)[3] , unsigned int (&end)[3] , void* pData);//Data size should match sizeof(Data type)*data length
    void update_mask(unsigned int (&begin)[3] , unsigned int (&end)[3] , unsigned char* pData);

private:
    void load_volume_resource_i();
    void release_volume_resource_i();
    void update_volume_resource_i();

private:
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<ImageData> _mask_data;
    std::shared_ptr<ImageDataHeader> _data_header;
    std::unique_ptr<BrickPool> m_pBrickPool;

    std::vector<GLTexture3DPtr> m_vecVolumeTex;//P.S here use vector for separate volume later
    std::vector<GLTexture3DPtr> m_vecMaskTex;

    GLBufferPtr m_pVolumeBrickInfoBuffer;
    std::map<LabelKey , GLBufferPtr> m_mapMaskBrickInfoBuffers;

    std::shared_ptr<CameraCalculator> _camera_calculator;

};

MED_IMAGING_END_NAMESPACE

#endif