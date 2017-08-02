#ifndef MED_IMG_BRICK_POOL_H_
#define MED_IMG_BRICK_POOL_H_


#include <memory>
#include <map>
#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"


MED_IMG_BEGIN_NAMESPACE

//GPU rendering use brick pool to accelerate
class ImageData;
class RenderAlgo_Export BrickPool
{
public:
    BrickPool();
    ~BrickPool();

    void set_volume(std::shared_ptr<ImageData> image_data);
    void set_mask(std::shared_ptr<ImageData> mask_data);

    void set_brick_size(unsigned int size);
    unsigned int get_brick_size() const;

    void set_brick_expand(unsigned int expand);
    unsigned int get_brick_expand() const;

    void get_brick_dim(unsigned int(&brick_dim)[3]);
    unsigned int get_brick_count() const;

    BrickCorner* get_brick_corner();

    VolumeBrickInfo* get_volume_brick_info();

    MaskBrickInfo* get_mask_brick_info(const std::vector<unsigned char>& vis_labels);

    void calculate_brick_corner();

    void calculate_brick_geometry();

    void calculate_volume_brick();

    void calculate_mask_brick();

    void update_mask_brick(unsigned int (&begin)[3] , unsigned int (&end)[3] , LabelKey label_key);

public:
    static void calculate_intercect_brick_index_range();

private:
    std::shared_ptr<ImageData> _volume;
    std::shared_ptr<ImageData> _mask;

    unsigned int _brick_size;
    unsigned int _brick_expand;
    unsigned int _brick_dim[3];
    unsigned int _brick_count;

    std::unique_ptr<BrickCorner[]> _brick_corner_array;
    BrickGeometry _brick_geometry;//For GL rendering

    std::unique_ptr<VolumeBrickInfo[]> _volume_brick_info_array;
    std::map<LabelKey , std::unique_ptr<MaskBrickInfo[]>> _mask_brick_info_array_set;

private:
    DISALLOW_COPY_AND_ASSIGN(BrickPool);
};

MED_IMG_END_NAMESPACE

#endif