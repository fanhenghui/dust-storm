#ifndef MED_IMAGING_BRICK_POOL_H_
#define MED_IMAGING_BRICK_POOL_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

//TODO dont need brick pool?
//Use brick info array to GPU&CPU
// VolumeInfos to contain volume & mask & volumebrick & maskbrick & volumebrickinfo & maskbrickinfo
MED_IMAGING_BEGIN_NAMESPACE

//TODO should not separate volume to bircks when GPU based 
class ImageData;
class RenderAlgo_Export BrickPool
{
public:
    BrickPool();

    ~BrickPool();

    void set_volume(std::shared_ptr<ImageData> image_data);

    void set_mask(std::shared_ptr<ImageData> image_data);

    void set_brick_size(unsigned int uiBrickSize);

    void set_brick_expand(unsigned int uiBrickExpand);

    void get_brick_dim(unsigned int (&uiBrickDim)[3]);

    BrickCorner* get_brick_corner();

    BrickUnit* get_volume_brick_unit();

    BrickUnit* get_mask_brick_unit();

    VolumeBrickInfo* get_volume_brick_info();

    MaskBrickInfo* get_mask_brick_info(const std::vector<unsigned char>& vecVisLabels);

    void calculate_volume_brick();

    void calculate_mask_brick();

    void update_mask_brick(unsigned int (&uiBegin)[3] , unsigned int (&uiEnd)[3]);

protected:

private:
    std::shared_ptr<ImageData> m_pVolume;
    std::shared_ptr<ImageData> m_pMask;

    unsigned int m_uiBrickSize;
    unsigned int m_uiBrickExpand;
    unsigned int m_uiBrickDim[3];

    std::unique_ptr<BrickCorner[]> m_pBrickCorner;
    std::unique_ptr<BrickUnit[]> m_pVolumeBrickUnit;
    std::unique_ptr<BrickUnit[]> m_pMaskBrickUnit;

    std::unique_ptr<VolumeBrickInfo[]> m_pVolumeBrickInfo;

    std::map<LabelKey , std::unique_ptr<MaskBrickInfo[]>> m_mapMaskBrickInfos;

    //TODO Brick cluster for VR entry&exit points rendering
};

MED_IMAGING_END_NAMESPACE

#endif