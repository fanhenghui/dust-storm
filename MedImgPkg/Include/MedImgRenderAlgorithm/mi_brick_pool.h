#ifndef MED_IMG_BRICK_POOL_H_
#define MED_IMG_BRICK_POOL_H_

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

//TODO dont need brick pool?
//Use brick info array to GPU&CPU
// VolumeInfos to contain volume & mask & volumebrick & maskbrick & volumebrickinfo & maskbrickinfo
MED_IMG_BEGIN_NAMESPACE

//TODO should not separate volume to bircks when GPU based 
class ImageData;
class RenderAlgo_Export BrickPool
{
public:
    BrickPool();

    ~BrickPool();

    void set_volume(std::shared_ptr<ImageData> image_data);

    void set_mask(std::shared_ptr<ImageData> image_data);

    void set_brick_size(unsigned int brick_size);

    void set_brick_expand(unsigned int brick_expand);

    void get_brick_dim(unsigned int (&brick_dim)[3]);

    BrickCorner* get_brick_corner();

    BrickUnit* get_volume_brick_unit();

    BrickUnit* get_mask_brick_unit();

    VolumeBrickInfo* get_volume_brick_info();

    MaskBrickInfo* get_mask_brick_info(const std::vector<unsigned char>& vis_labels);

    void calculate_volume_brick();

    void calculate_mask_brick();

    void update_mask_brick(unsigned int (&begin)[3] , unsigned int (&end)[3]);

protected:

private:
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<ImageData> _mask_data;

    unsigned int _brick_size;
    unsigned int _brick_expand;
    unsigned int _brick_dim[3];

    std::unique_ptr<BrickCorner[]> _brick_corner_array;
    std::unique_ptr<BrickUnit[]> _volume_brick_unit_array;
    std::unique_ptr<BrickUnit[]> _mask_brick_unit_array;

    std::unique_ptr<VolumeBrickInfo[]> _volume_brick_info_array;

    std::map<LabelKey , std::unique_ptr<MaskBrickInfo[]>> _mask_brick_info_array_set;

    //TODO Brick cluster for VR entry&exit points rendering
};

MED_IMG_END_NAMESPACE

#endif