#include "mi_ray_casting_gpu.h"

#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_vao.h"
#include "MedImgGLResource/mi_gl_buffer.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"
#include "mi_rc_step_base.h"
#include "mi_rc_step_main.h"
#include "mi_rc_step_composite.h"
#include "mi_rc_step_ray_casting.h"
#include "mi_rc_step_color_inverse.h"
#include "mi_rc_step_mask_sampler.h"
#include "mi_rc_step_volume_sampler.h"
#include "mi_rc_step_utils.h"
#include "mi_rc_step_shading.h"

MED_IMAGING_BEGIN_NAMESPACE

RayCastingGPU::RayCastingGPU(std::shared_ptr<RayCaster> ray_caster):_ray_caster(ray_caster)
{

}

RayCastingGPU::~RayCastingGPU()
{

}

void RayCastingGPU::render(int test_code)
{
    update_i();

    CHECK_GL_ERROR;

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(false);

    _gl_vao->bind();
    _program->bind();

    for (auto it = _ray_casting_steps.begin() ; it != _ray_casting_steps.end() ; ++it)
    {
        (*it)->set_gpu_parameter();
    }

    glDrawArrays(GL_TRIANGLES , 0 , 6);

    _gl_vao->unbind();
    _program->unbind();
    glPopAttrib();

    CHECK_GL_ERROR;
}

void RayCastingGPU::update_i()
{
    CHECK_GL_ERROR;

    //Create VAO
    if (!_gl_vao)
    {
        UIDType uid;
        _gl_vao = GLResourceManagerContainer::instance()->get_vao_manager()->create_object(uid);
        _gl_vao->set_description("Ray casting GPU VAO");
        _gl_vao->initialize();

        _gl_buffer_vertex = GLResourceManagerContainer::instance()->get_buffer_manager()->create_object(uid);
        _gl_buffer_vertex->set_description("Ray casting GPU vertex buffer (-1 -1)~ (1,1)");
        _gl_buffer_vertex->initialize();
        _gl_buffer_vertex->set_buffer_target(GL_ARRAY_BUFFER);

        _gl_vao->bind();
        _gl_buffer_vertex->bind();
        float vertex[] = {-1,1,0 , 
            -1,-1,0 , 
            1,-1,0,
            1,-1,0,
            1,1,0,
            -1,1,0 };
        _gl_buffer_vertex->load(sizeof(vertex) , vertex , GL_STATIC_DRAW);
        glVertexAttribPointer(0 , 3 , GL_FLOAT , GL_FALSE , 0 , NULL );
        glEnableVertexAttribArray(0);

        _gl_vao->unbind();
        _gl_buffer_vertex->unbind();
    }

    CHECK_GL_ERROR;

    //Create Program
    if (!_program)
    {
        UIDType uid;
        _program = GLResourceManagerContainer::instance()->get_program_manager()->create_object(uid);
        _program->set_description("Ray casting GPU program");
        _program->initialize();
    }

    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();

    if (_ray_casting_steps.empty() || 
        _mask_mode != ray_caster->_mask_mode ||
        _composite_mode != ray_caster->_composite_mode||
        _interpolation_mode != ray_caster->_interpolation_mode||
        _shading_mode != ray_caster->_shading_mode ||
        _color_inverse_mode != ray_caster->_color_inverse_mode)
    {
        _ray_casting_steps.clear();
        _mask_mode = ray_caster->_mask_mode;
        _composite_mode = ray_caster->_composite_mode;
        _interpolation_mode = ray_caster->_interpolation_mode;
        _shading_mode = ray_caster->_shading_mode;
        _color_inverse_mode = ray_caster->_color_inverse_mode;

#define STEP_PUSH_BACK(step_class_name) \
        _ray_casting_steps.push_back(std::shared_ptr<step_class_name>(new step_class_name(ray_caster , _program)));

        //Main
        STEP_PUSH_BACK(RCStepMainVert);
        STEP_PUSH_BACK(RCStepMainFrag);

        //Utils
        STEP_PUSH_BACK(RCStepUtils);

        //Composite
        if (_composite_mode == COMPOSITE_DVR)
        {

        }
        else if (_composite_mode == COMPOSITE_AVERAGE)
        {
            STEP_PUSH_BACK(RCStepRayCastingAverage);
            STEP_PUSH_BACK(RCStepCompositeAverage);
        }
        else if (_composite_mode == COMPOSITE_MIP)
        {
            STEP_PUSH_BACK(RCStepRayCastingMIPMinIP);
            STEP_PUSH_BACK(RCStepCompositeMIP);
        }
        else if (_composite_mode == COMPOSITE_MINIP)
        {
            STEP_PUSH_BACK(RCStepRayCastingMIPMinIP);
            STEP_PUSH_BACK(RCStepCompositeMinIP);
        }

        //Mask
        if (_mask_mode == MASK_NONE)
        {
            STEP_PUSH_BACK(RCStepMaskNoneSampler);
        }
        else if (_mask_mode == MASK_MULTI_LABEL)
        {
            STEP_PUSH_BACK(RCStepMaskNearstSampler);
        }

        //Volume 
        if (_interpolation_mode == LINEAR)
        {
            STEP_PUSH_BACK(RCStepVolumeLinearSampler);
        }
        else if (_interpolation_mode == NEARST)
        {
            STEP_PUSH_BACK(RCStepVolumeNearstSampler);
        }
        else if (_interpolation_mode == CUBIC)
        {
            //TODO
        }

        //Shading
        if (_shading_mode == SHADING_NONE)
        {
            STEP_PUSH_BACK(RCStepShadingNone);
        }
        else if (_shading_mode == SHADING_PHONG)
        {
            STEP_PUSH_BACK(RCStepShadingPhong);
        }

        //Color inverse
        if (_color_inverse_mode == COLOR_INVERSE_DISABLE)
        {
            STEP_PUSH_BACK(RCStepColorInverseDisable);
        }
        else //(_color_inverse_mode == COLOR_INVERSE_ENABLE)
        {
            STEP_PUSH_BACK(RCStepColorInverseEnable);
        }

#undef STEP_PUSH_BACK



        //compile
        std::vector<GLShaderInfo> shaders;
        for (auto it = _ray_casting_steps.begin() ; it != _ray_casting_steps.end() ; ++it)
        {
            shaders.push_back((*it)->get_shader_info());
        }
        _program->set_shaders(shaders);
        _program->compile();

        for (auto it = _ray_casting_steps.begin() ; it != _ray_casting_steps.end() ; ++it)
        {
            (*it)->get_uniform_location();
        }
    }

    CHECK_GL_ERROR;
}

MED_IMAGING_END_NAMESPACE