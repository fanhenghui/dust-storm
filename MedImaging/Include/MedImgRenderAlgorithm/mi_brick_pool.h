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

    void SetVolume(std::shared_ptr<ImageData> pImgData);

    void SetMask(std::shared_ptr<ImageData> pImgData);

    void SetBrickSize(unsigned int uiBrickSize);

    void SetBrickExpand(unsigned int uiBrickExpand);

    void GetBrickDim(unsigned int (&uiBrickDim)[3]);

    BrickCorner* GetBrickCorner();

    BrickUnit* GetVolumeBrickUnit();

    BrickUnit* GetMaskBrickUnit();

    VolumeBrickInfo* GetVolumeBrickInfo();

    MaskBrickInfo* GetMaskBrickInfo(const std::vector<unsigned char>& vecVisLabels);

    void CalculateVolumeBrick();

    void CalculateMaskBrick();

    void UpdateMaskBrick(unsigned int (&uiBegin)[3] , unsigned int (&uiEnd)[3]);

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