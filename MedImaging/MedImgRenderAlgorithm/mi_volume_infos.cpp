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

void VolumeInfos::finialize()
{
    release_volume_resource_i();
}

void VolumeInfos::set_volume(std::shared_ptr<ImageData> image_data)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(image_data);

        _volume_data = image_data;

        //////////////////////////////////////////////////////////////////////////
        //Create camera calculator
        _camera_calculator.reset( new CameraCalculator(_volume_data));

        //////////////////////////////////////////////////////////////////////////
        //Create brick pool & calculate volume brick
        //TODO brick info used in GPU , but brick unit useless 
        //////////////////////////////////////////////////////////////////////////
        //m_pBrickPool.reset(new BrickPool());//Reset brick pool
        //m_pBrickPool->set_volume(_volume_data);
        //m_pBrickPool->set_brick_size(BrickUtils::instance()->GetBrickSize());
        //m_pBrickPool->set_brick_expand(BrickUtils::instance()->get_brick_expand());

        //m_pBrickPool->calculate_volume_brick();

        //////////////////////////////////////////////////////////////////////////
        //Upload volume texture(TODO using data loader to wrap it . for separate volume later)
        //release previous resource
        release_volume_resource_i();

        //////////////////////////////////////////////////////////////////////////
        //Upload volume brick info
        load_volume_resource_i();

    }
    catch (const Exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        assert(false);
    }
}

void VolumeInfos::set_mask(std::shared_ptr<ImageData> image_data)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(image_data);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pBrickPool);

        _mask_data = image_data;
        m_pBrickPool->set_mask(_mask_data);
        m_pBrickPool->calculate_mask_brick();
    }
    catch (const Exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        assert(false);
    }
}

void VolumeInfos::set_data_header(std::shared_ptr<ImageDataHeader> data_header)
{
    _data_header = data_header;
}

std::vector<GLTexture3DPtr> VolumeInfos::get_volume_texture()
{
    return m_vecVolumeTex;
}

std::vector<GLTexture3DPtr> VolumeInfos::get_mask_texture()
{
    return m_vecMaskTex;
}

GLBufferPtr VolumeInfos::get_volume_brick_info_buffer()
{
    return m_pVolumeBrickInfoBuffer;
}

//GLBufferPtr VolumeInfos::GetMaskBrickInfoBuffer(const std::vector<unsigned char>& vis_labels)
//{
//    //TODO check dirty
//}

std::shared_ptr<ImageData> VolumeInfos::get_volume()
{
    return _volume_data;
}

std::shared_ptr<ImageData> VolumeInfos::get_mask()
{
    return _mask_data;
}

BrickCorner* VolumeInfos::get_brick_corner()
{
    return m_pBrickPool->get_brick_corner();
}

BrickUnit* VolumeInfos::get_volume_brick_unit()
{
    return m_pBrickPool->get_volume_brick_unit();
}

BrickUnit* VolumeInfos::get_mask_brick_unit()
{
    return m_pBrickPool->get_mask_brick_unit();
}

VolumeBrickInfo* VolumeInfos::get_volume_brick_info()
{
    return m_pBrickPool->get_volume_brick_info();
}

MaskBrickInfo* VolumeInfos::get_mask_brick_info(const std::vector<unsigned char>& vis_labels)
{
    return m_pBrickPool->get_mask_brick_info(vis_labels);
}

void VolumeInfos::update_volume(unsigned int (&begin)[3] , unsigned int (&end)[3] , void* pData)
{
    //TODO
    //update volume CPU

    //update volume GPU
}

void VolumeInfos::update_mask(unsigned int (&begin)[3] , unsigned int (&end)[3] , unsigned char* pData)
{
    //TODO
    //update mask CPU

    //update mask GPU
}

void VolumeInfos::release_volume_resource_i()
{
    //release volume textures
    if (!m_vecVolumeTex.empty())
    {
        for (auto it = m_vecVolumeTex.begin() ; it != m_vecVolumeTex.end() ; ++it)
        {
            GLResourceManagerContainer::instance()->get_texture_3d_manager()->remove_object((*it)->get_uid());
        }

        m_vecVolumeTex.clear();
        GLResourceManagerContainer::instance()->get_texture_3d_manager()->update();
    }

    //release volume brick info
    if (m_pVolumeBrickInfoBuffer)
    {
        GLResourceManagerContainer::instance()->get_buffer_manager()->remove_object(m_pVolumeBrickInfoBuffer->get_uid());
        m_pVolumeBrickInfoBuffer.reset();
        GLResourceManagerContainer::instance()->get_buffer_manager()->update();
    }
}

void VolumeInfos::load_volume_resource_i()
{
    if (GPU == Configuration::instance()->get_processing_unit_type())
    {
        //////////////////////////////////////////////////////////////////////////
        // 1 Volume texture
        m_vecVolumeTex.clear();

        //Single volume
        UIDType uid(0);
        GLTexture3DPtr pTex = GLResourceManagerContainer::instance()->get_texture_3d_manager()->create_object(uid);
        if (_data_header)
        {
            pTex->set_description("Volume : " + _data_header->series_uid);
        }
        else
        {
            pTex->set_description("Volume : Undefined series UID");
        }

        pTex->initialize();
        pTex->bind();
        GLTextureUtils::set_1d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_3D , GL_LINEAR);
        GLenum internal_format , format , eType;
        GLUtils::get_gray_texture_format(_volume_data->_data_type , internal_format , format ,eType);
        pTex->load(internal_format ,_volume_data->_dim[0] , _volume_data->_dim[1] , _volume_data->_dim[2] , format, eType , _volume_data->get_pixel_pointer());
        pTex->unbind();

        m_vecVolumeTex.push_back(pTex);

        //TODO separate volumes

        //////////////////////////////////////////////////////////////////////////
        //2 Volume brick info
        /*m_pVolumeBrickInfoBuffer = GLResourceManagerContainer::instance()->get_buffer_manager()->create_object(uid);
        m_pVolumeBrickInfoBuffer->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
        m_pVolumeBrickInfoBuffer->initialize();
        m_pVolumeBrickInfoBuffer->bind();

        unsigned int brick_dim[3] = {1,1,1};
        m_pBrickPool->get_brick_dim(brick_dim);
        const unsigned int brick_count = brick_dim[0]*brick_dim[1]*brick_dim[2];
        m_pVolumeBrickInfoBuffer->load(brick_count*sizeof(VolumeBrickInfo) , m_pBrickPool->get_volume_brick_info() , GL_STATIC_DRAW);
        m_pVolumeBrickInfoBuffer->unbind();*/
    }
}

void VolumeInfos::update_volume_resource_i()
{
    if (GPU == Configuration::instance()->get_processing_unit_type())
    {
        //TODO update
    }
}

std::shared_ptr<ImageDataHeader> VolumeInfos::get_data_header()
{
    return _data_header;
}

std::shared_ptr<CameraCalculator> VolumeInfos::get_camera_calculator()
{
    return _camera_calculator;
}

MED_IMAGING_END_NAMESPACE