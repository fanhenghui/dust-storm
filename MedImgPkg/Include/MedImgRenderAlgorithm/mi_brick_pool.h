#ifndef MED_IMG_BRICK_POOL_H_
#define MED_IMG_BRICK_POOL_H_


#include <memory>
#include <map>
#include <string>

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgArithmetic/mi_aabb.h"

MED_IMG_BEGIN_NAMESPACE

//GPU rendering use brick pool to accelerate
class ImageData;
class VolumeBrickInfoCalculator;
class MaskBrickInfoCalculator;
class RenderAlgo_Export BrickPool
{
public:
    BrickPool(unsigned int brick_size , unsigned int brick_margin);
    ~BrickPool();

    void set_volume(std::shared_ptr<ImageData> image_data);
    void set_mask(std::shared_ptr<ImageData> mask_data);

    void set_volume_texture(GLTexture3DPtr tex);
    void set_mask_texture(GLTexture3DPtr tex);

    unsigned int get_brick_size() const;
    unsigned int get_brick_margin() const;

    void get_brick_dim(unsigned int(&brick_dim)[3]);
    unsigned int get_brick_count() const;

    void calculate_brick_geometry();
    const BrickGeometry& get_brick_geometry() const;

    void calculate_volume_brick_info();
    VolumeBrickInfo* get_volume_brick_info() const;
    void write_volume_brick_info(const std::string& path);

    void calculate_mask_brick_info(const std::vector<unsigned char>& vis_labels);
    void update_mask_brick_info(const AABBUI& aabb);
    MaskBrickInfo* get_mask_brick_info(const std::vector<unsigned char>& vis_labels) const;
    void write_mask_brick_info(const std::string& path , const std::vector<unsigned char>& visible_labels);

    void remove_mask_brick_info(const std::vector<unsigned char>& vis_labels);
    void remove_all_mask_brick_info();

public:
    void calculate_intercect_brick_range(const AABB& bounding , AABBI& brick_range);
    void get_clipping_brick_geometry(const AABB& bounding, float* brick_vertex, float* brick_color);

private:
    std::shared_ptr<ImageData> _volume;
    std::shared_ptr<ImageData> _mask;
    GLTexture3DPtr _volume_texture;
    GLTexture3DPtr _mask_texture;

    unsigned int _brick_size;
    unsigned int _brick_margin;
    unsigned int _brick_dim[3];
    unsigned int _brick_count;

    BrickGeometry _brick_geometry;//For GL rendering

    std::unique_ptr<VolumeBrickInfo[]> _volume_brick_info_array;
    GLBufferPtr _volume_brick_info_buffer;

    std::map<LabelKey , std::unique_ptr<MaskBrickInfo[]>> _mask_brick_info_array_set;
    std::map<LabelKey , GLBufferPtr> _mask_brick_info_buffer_set;

    GLResourceShield _res_shield;

private:
    std::unique_ptr<VolumeBrickInfoCalculator> _volume_brick_info_cal;
    std::unique_ptr<MaskBrickInfoCalculator> _mask_brick_info_cal;

private:
    DISALLOW_COPY_AND_ASSIGN(BrickPool);
};

MED_IMG_END_NAMESPACE

#endif