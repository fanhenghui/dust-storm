#include "mi_rc_step_main.h"
#include "mi_shader_collection.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgIO/mi_image_data.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"

MED_IMG_BEGIN_NAMESPACE


GLShaderInfo RCStepMainVert::get_shader_info()
{
    return GLShaderInfo(GL_VERTEX_SHADER , S_RC_MAIN_VERTEX , "RCStepMainVert");
}

void RCStepMainVert::set_gpu_parameter()
{
}

GLShaderInfo RCStepMainFrag::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_MAIN_FRAGMENT , "RCStepMainFrag");
}

void RCStepMainFrag::set_gpu_parameter()
{
    CHECK_GL_ERROR;

    GLProgramPtr program = _gl_program.lock();
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
    int act_tex = _act_tex_counter->tick();
    glActiveTexture(GL_TEXTURE0 + act_tex);
    volume_textures[0]->bind();
    GLTextureUtils::set_3d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_3D , GL_LINEAR);
    glUniform1i(_loc_volume_data , act_tex);
    glDisable(GL_TEXTURE_3D);

    //3 Volume dimension
    glUniform3f(_loc_volume_dim , (float)volume_img->_dim[0] , 
        (float)volume_img->_dim[1] , (float)volume_img->_dim[2]);

    //4 Sample rate
    glUniform1f(_loc_sample_rate , ray_caster->get_sample_rate());

    //5 Mask texture
    std::vector<GLTexture3DPtr> mask_textures = ray_caster->get_mask_data_texture();
    if (!mask_textures.empty())
    {
        glEnable(GL_TEXTURE_3D);
        act_tex = _act_tex_counter->tick();
        glActiveTexture(GL_TEXTURE0 + act_tex);
        mask_textures[0]->bind();
        GLTextureUtils::set_3d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_3D , GL_NEAREST);
        glUniform1i(_loc_mask_data , act_tex);
        glDisable(GL_TEXTURE_3D);
    }


    CHECK_GL_ERROR;
}

void RCStepMainFrag::get_uniform_location()
{
    GLProgramPtr program = _gl_program.lock();
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

GLShaderInfo RCStepMainTestFrag::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_MAIN_TEST_FRAGMENT , "RCStepMainTestFrag");
}

void RCStepMainTestFrag::set_gpu_parameter()
{
    CHECK_GL_ERROR;

    GLProgramPtr program = _gl_program.lock();
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

    //2 Volume dimension
    glUniform3f(_loc_volume_dim , (float)volume_img->_dim[0] , 
        (float)volume_img->_dim[1] , (float)volume_img->_dim[2]);

    //3 Test code
    glUniform1i(_loc_test_code , ray_caster->get_test_code());


    CHECK_GL_ERROR;
}

void RCStepMainTestFrag::get_uniform_location()
{
    GLProgramPtr program = _gl_program.lock();
    _loc_volume_dim = program->get_uniform_location("volume_dim");
    _loc_test_code = program->get_uniform_location("test_code");

    if (-1 == _loc_volume_dim ||
        -1 == _loc_test_code)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}



MED_IMG_END_NAMESPACE