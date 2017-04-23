#include "mi_rc_step_main.h"
#include "mi_shader_collection.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgIO/mi_image_data.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"

MED_IMAGING_BEGIN_NAMESPACE


GLShaderInfo RCStepMainVert::get_shader_info()
{
    
    return GLShaderInfo(GL_VERTEX_SHADER , ksRCMainVert , "RCStepMainVert");
}

void RCStepMainVert::set_gpu_parameter()
{
}

GLShaderInfo RCStepMainFrag::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCMainFrag , "RCStepMainFrag");
}

void RCStepMainFrag::set_gpu_parameter()
{
    CHECK_GL_ERROR;

    GLProgramPtr program = _program.lock();
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    std::shared_ptr<ImageData> volume_img = ray_caster->get_volume_data();

    RENDERALGO_CHECK_NULL_EXCEPTION(volume_img);

    //1 Entry exit points
    std::shared_ptr<EntryExitPoints> entry_exit_points = ray_caster->get_entry_exit_points();
    RENDERALGO_CHECK_NULL_EXCEPTION(entry_exit_points);

    GLTexture2DPtr entry_texture = entry_exit_points->get_entry_points_texture();
    GLTexture2DPtr exit_texture = entry_exit_points->get_exit_points_texture();


#define IMG_BINDING_ENTRY_POINTS  0
#define IMG_BINDING_EXIT_POINTS  1

    entry_texture->bind_image(IMG_BINDING_ENTRY_POINTS , 0 , GL_FALSE , 0 , GL_READ_ONLY , GL_RGBA32F);
    exit_texture->bind_image(IMG_BINDING_EXIT_POINTS , 0 , GL_FALSE , 0 , GL_READ_ONLY , GL_RGBA32F);


#undef IMG_BINDING_ENTRY_POINTS
#undef IMG_BINDING_EXIT_POINTS

    //2 Volume texture
    std::vector<GLTexture3DPtr> volume_textures = ray_caster->get_volume_data_texture();
    if (volume_textures.empty())
    {
        RENDERALGO_THROW_EXCEPTION("Volume texture is empty!");
    }
    glEnable(GL_TEXTURE_3D);
    glActiveTexture(GL_TEXTURE1);
    volume_textures[0]->bind();
    GLTextureUtils::set_1d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_3D , GL_LINEAR);
    glUniform1i(_loc_volume_data , 1);
    glDisable(GL_TEXTURE_3D);

    //3 Volume dimension
    glUniform3f(_loc_volume_dim , (float)volume_img->_dim[0] , 
        (float)volume_img->_dim[1] , (float)volume_img->_dim[2]);

    //4 Sample rate
    glUniform1f(_loc_sample_rate , ray_caster->get_sample_rate());

    //TODO Mask related



    CHECK_GL_ERROR;
}

void RCStepMainFrag::get_uniform_location()
{
    GLProgramPtr program = _program.lock();
    _loc_volume_dim = program->get_uniform_location("volume_dim");
    _loc_volume_data = program->get_uniform_location("volume_sampler");
    _loc_mask_data = program->get_uniform_location("mask_sampler");
    _loc_sample_rate = program->get_uniform_location("sample_rate");

    if (-1 == _loc_volume_dim ||
        -1 == _loc_volume_data ||
        //-1 == m_iLocMaskData ||
        -1 == _loc_sample_rate)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}



MED_IMAGING_END_NAMESPACE