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

    void Finialize();

    //Set input
    void SetVolume(std::shared_ptr<ImageData> pImgData);
    void SetMask(std::shared_ptr<ImageData> pImgData);
    void SetDataHeader(std::shared_ptr<ImageDataHeader> pDataHeader);

    //Get camera calculator
    std::shared_ptr<CameraCalculator> GetCameraCalculator();

    //Get GPU resource
    std::vector<GLTexture3DPtr> GetVolumeTexture();
    std::vector<GLTexture3DPtr> GetMaskTexture();

    GLBufferPtr GetVolumeBrickInfoBuffer();
    //GLBufferPtr GetMaskBrickInfoBuffer(const std::vector<unsigned char>& vecVisLabels);

    //Get GPU resource
    std::shared_ptr<ImageData> GetVolume();
    std::shared_ptr<ImageData> GetMask();
    std::shared_ptr<ImageDataHeader> GetDataHeader();

    BrickCorner* GetBrickCorner();
    BrickUnit* GetVolumeBrickUnit();
    BrickUnit* GetMaskBrickUnit();
    VolumeBrickInfo* GetVolumeBrickInfo();
    MaskBrickInfo* GetMaskBrickInfo(const std::vector<unsigned char>& vecVisLabels);

    //Update(should update to CPU and GPU)
    void UpdateVolume(unsigned int (&uiBegin)[3] , unsigned int (&uiEnd)[3] , void* pData);//Data size should match sizeof(Data type)*data length
    void UpdateMask(unsigned int (&uiBegin)[3] , unsigned int (&uiEnd)[3] , unsigned char* pData);

private:
    void LoadVolumeResource_i();
    void ReleaseVolumeResource_i();
    void UpdateVolumeResource_i();

private:
    std::shared_ptr<ImageData> m_pVolume;
    std::shared_ptr<ImageData> m_pMask;
    std::shared_ptr<ImageDataHeader> m_pDataHeader;
    std::unique_ptr<BrickPool> m_pBrickPool;

    std::vector<GLTexture3DPtr> m_vecVolumeTex;//P.S here use vector for separate volume later
    std::vector<GLTexture3DPtr> m_vecMaskTex;

    GLBufferPtr m_pVolumeBrickInfoBuffer;
    std::map<LabelKey , GLBufferPtr> m_mapMaskBrickInfoBuffers;

    std::shared_ptr<CameraCalculator> m_pCameraCalculator;

};

MED_IMAGING_END_NAMESPACE

#endif