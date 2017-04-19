#include "mi_volume_infos.h"

#include "MedImgCommon/mi_configuration.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_buffer.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_camera_calculator.h"
#include "mi_brick_generator.h"
#include "mi_brick_info_generator.h"
#include "mi_brick_pool.h"
#include "mi_brick_utils.h"

MED_IMAGING_BEGIN_NAMESPACE


VolumeInfos::VolumeInfos()
{

}

VolumeInfos::~VolumeInfos()
{
    
}

void VolumeInfos::Finialize()
{
    ReleaseVolumeResource_i();
}

void VolumeInfos::SetVolume(std::shared_ptr<ImageData> pImgData)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(pImgData);

        m_pVolume = pImgData;

        //////////////////////////////////////////////////////////////////////////
        //Create camera calculator
        m_pCameraCalculator.reset( new CameraCalculator(m_pVolume));

        //////////////////////////////////////////////////////////////////////////
        //Create brick pool & calculate volume brick
        //TODO brick info used in GPU , but brick unit useless 
        //////////////////////////////////////////////////////////////////////////
        //m_pBrickPool.reset(new BrickPool());//Reset brick pool
        //m_pBrickPool->SetVolume(m_pVolume);
        //m_pBrickPool->SetBrickSize(BrickUtils::Instance()->GetBrickSize());
        //m_pBrickPool->SetBrickExpand(BrickUtils::Instance()->GetBrickExpand());

        //m_pBrickPool->CalculateVolumeBrick();

        //////////////////////////////////////////////////////////////////////////
        //Upload volume texture(TODO using data loader to wrap it . for separate volume later)
        //Release previous resource
        ReleaseVolumeResource_i();

        //////////////////////////////////////////////////////////////////////////
        //Upload volume brick info
        LoadVolumeResource_i();

    }
    catch (const Exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        assert(false);
    }
}

void VolumeInfos::SetMask(std::shared_ptr<ImageData> pImgData)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(pImgData);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pBrickPool);

        m_pMask = pImgData;
        m_pBrickPool->SetMask(m_pMask);
        m_pBrickPool->CalculateMaskBrick();
    }
    catch (const Exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        assert(false);
    }
}

void VolumeInfos::SetDataHeader(std::shared_ptr<ImageDataHeader> pDataHeader)
{
    m_pDataHeader = pDataHeader;
}

std::vector<GLTexture3DPtr> VolumeInfos::GetVolumeTexture()
{
    return m_vecVolumeTex;
}

std::vector<GLTexture3DPtr> VolumeInfos::GetMaskTexture()
{
    return m_vecMaskTex;
}

GLBufferPtr VolumeInfos::GetVolumeBrickInfoBuffer()
{
    return m_pVolumeBrickInfoBuffer;
}

//GLBufferPtr VolumeInfos::GetMaskBrickInfoBuffer(const std::vector<unsigned char>& vecVisLabels)
//{
//    //TODO check dirty
//}

std::shared_ptr<ImageData> VolumeInfos::GetVolume()
{
    return m_pVolume;
}

std::shared_ptr<ImageData> VolumeInfos::GetMask()
{
    return m_pMask;
}

BrickCorner* VolumeInfos::GetBrickCorner()
{
    return m_pBrickPool->GetBrickCorner();
}

BrickUnit* VolumeInfos::GetVolumeBrickUnit()
{
    return m_pBrickPool->GetVolumeBrickUnit();
}

BrickUnit* VolumeInfos::GetMaskBrickUnit()
{
    return m_pBrickPool->GetMaskBrickUnit();
}

VolumeBrickInfo* VolumeInfos::GetVolumeBrickInfo()
{
    return m_pBrickPool->GetVolumeBrickInfo();
}

MaskBrickInfo* VolumeInfos::GetMaskBrickInfo(const std::vector<unsigned char>& vecVisLabels)
{
    return m_pBrickPool->GetMaskBrickInfo(vecVisLabels);
}

void VolumeInfos::UpdateVolume(unsigned int (&uiBegin)[3] , unsigned int (&uiEnd)[3] , void* pData)
{
    //TODO
    //Update volume CPU

    //Update volume GPU
}

void VolumeInfos::UpdateMask(unsigned int (&uiBegin)[3] , unsigned int (&uiEnd)[3] , unsigned char* pData)
{
    //TODO
    //Update mask CPU

    //Update mask GPU
}

void VolumeInfos::ReleaseVolumeResource_i()
{
    //Release volume textures
    if (!m_vecVolumeTex.empty())
    {
        for (auto it = m_vecVolumeTex.begin() ; it != m_vecVolumeTex.end() ; ++it)
        {
            GLResourceManagerContainer::Instance()->GetTexture3DManager()->RemoveObject((*it)->GetUID());
        }

        m_vecVolumeTex.clear();
        GLResourceManagerContainer::Instance()->GetTexture3DManager()->Update();
    }

    //Release volume brick info
    if (m_pVolumeBrickInfoBuffer)
    {
        GLResourceManagerContainer::Instance()->GetBufferManager()->RemoveObject(m_pVolumeBrickInfoBuffer->GetUID());
        m_pVolumeBrickInfoBuffer.reset();
        GLResourceManagerContainer::Instance()->GetBufferManager()->Update();
    }
}

void VolumeInfos::LoadVolumeResource_i()
{
    if (GPU == Configuration::Instance()->GetProcessingUnitType())
    {
        //////////////////////////////////////////////////////////////////////////
        // 1 Volume texture
        m_vecVolumeTex.clear();

        //Single volume
        UIDType uid(0);
        GLTexture3DPtr pTex = GLResourceManagerContainer::Instance()->GetTexture3DManager()->CreateObject(uid);
        if (m_pDataHeader)
        {
            pTex->SetDescription("Volume : " + m_pDataHeader->m_sSeriesUID);
        }
        else
        {
            pTex->SetDescription("Volume : Undefined series UID");
        }

        pTex->Initialize();
        pTex->Bind();
        GLTextureUtils::Set3DWrapSTR(GL_CLAMP_TO_BORDER);
        GLTextureUtils::SetFilter(GL_TEXTURE_3D , GL_LINEAR);
        GLenum eInternalFormat , eFormat , eType;
        GLUtils::GetGrayTextureFormat(m_pVolume->m_eDataType , eInternalFormat , eFormat ,eType);
        pTex->Load(eInternalFormat ,m_pVolume->m_uiDim[0] , m_pVolume->m_uiDim[1] , m_pVolume->m_uiDim[2] , eFormat, eType , m_pVolume->GetPixelPointer());
        pTex->UnBind();

        m_vecVolumeTex.push_back(pTex);

        //TODO separate volumes

        //////////////////////////////////////////////////////////////////////////
        //2 Volume brick info
        /*m_pVolumeBrickInfoBuffer = GLResourceManagerContainer::Instance()->GetBufferManager()->CreateObject(uid);
        m_pVolumeBrickInfoBuffer->SetBufferTarget(GL_SHADER_STORAGE_BUFFER);
        m_pVolumeBrickInfoBuffer->Initialize();
        m_pVolumeBrickInfoBuffer->Bind();

        unsigned int uiBrickDim[3] = {1,1,1};
        m_pBrickPool->GetBrickDim(uiBrickDim);
        const unsigned int uiBrickCount = uiBrickDim[0]*uiBrickDim[1]*uiBrickDim[2];
        m_pVolumeBrickInfoBuffer->Load(uiBrickCount*sizeof(VolumeBrickInfo) , m_pBrickPool->GetVolumeBrickInfo() , GL_STATIC_DRAW);
        m_pVolumeBrickInfoBuffer->UnBind();*/
    }
}

void VolumeInfos::UpdateVolumeResource_i()
{
    if (GPU == Configuration::Instance()->GetProcessingUnitType())
    {
        //TODO Update
    }
}

std::shared_ptr<ImageDataHeader> VolumeInfos::GetDataHeader()
{
    return m_pDataHeader;
}

std::shared_ptr<CameraCalculator> VolumeInfos::GetCameraCalculator()
{
    return m_pCameraCalculator;
}

MED_IMAGING_END_NAMESPACE