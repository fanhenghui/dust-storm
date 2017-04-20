#include "mi_ray_casting_gpu.h"
#include "mi_ray_caster.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_vao.h"
#include "MedImgGLResource/mi_gl_buffer.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_utils.h"

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

RayCastingGPU::RayCastingGPU(std::shared_ptr<RayCaster> pRayCaster):m_pRayCaster(pRayCaster)
{

}

RayCastingGPU::~RayCastingGPU()
{

}

void RayCastingGPU::render(int iTestCode)
{
    update_i();

    CHECK_GL_ERROR;

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(false);

    m_pVAO->bind();
    m_pProgram->bind();

    for (auto it = m_vecSteps.begin() ; it != m_vecSteps.end() ; ++it)
    {
        (*it)->set_gpu_parameter();
    }

    glDrawArrays(GL_TRIANGLES , 0 , 6);

    m_pVAO->unbind();
    m_pProgram->unbind();
    glPopAttrib();

    CHECK_GL_ERROR;
}

void RayCastingGPU::update_i()
{
    CHECK_GL_ERROR;

    //Create VAO
    if (!m_pVAO)
    {
        UIDType uid;
        m_pVAO = GLResourceManagerContainer::instance()->get_vao_manager()->create_object(uid);
        m_pVAO->set_description("Ray casting GPU VAO");
        m_pVAO->initialize();

        m_pVertexBuffer = GLResourceManagerContainer::instance()->get_buffer_manager()->create_object(uid);
        m_pVertexBuffer->set_description("Ray casting GPU vertex buffer (-1 -1)~ (1,1)");
        m_pVertexBuffer->initialize();
        m_pVertexBuffer->set_buffer_target(GL_ARRAY_BUFFER);

        m_pVAO->bind();
        m_pVertexBuffer->bind();
        float pVertex[] = { -1,1,0 , -1,-1,0 , 1,-1,0,
        1,-1,0, 1,1,0,-1,1,0 };
        m_pVertexBuffer->load(sizeof(pVertex) , pVertex , GL_STATIC_DRAW);
        glVertexAttribPointer(0 , 3 , GL_FLOAT , GL_FALSE , 0 , NULL );
        glEnableVertexAttribArray(0);

        m_pVAO->unbind();
        m_pVertexBuffer->unbind();
    }

    CHECK_GL_ERROR;

    //Create Program
    if (!m_pProgram)
    {
        UIDType uid;
        m_pProgram = GLResourceManagerContainer::instance()->get_program_manager()->create_object(uid);
        m_pProgram->set_description("Ray casting GPU program");
        m_pProgram->initialize();
    }

    std::shared_ptr<RayCaster> pRayCaster = m_pRayCaster.lock();

    if (m_vecSteps.empty() || 
        m_eMaskMode != pRayCaster->m_eMaskMode ||
        m_eCompositeMode != pRayCaster->m_eCompositeMode||
        m_eInterpolationMode != pRayCaster->m_eInterpolationMode||
        m_eShadingMode != pRayCaster->m_eShadingMode ||
        m_eColorInverseMode != pRayCaster->m_eColorInverseMode)
    {
        m_vecSteps.clear();
        m_eMaskMode = pRayCaster->m_eMaskMode;
        m_eCompositeMode = pRayCaster->m_eCompositeMode;
        m_eInterpolationMode = pRayCaster->m_eInterpolationMode;
        m_eShadingMode = pRayCaster->m_eShadingMode;
        m_eColorInverseMode = pRayCaster->m_eColorInverseMode;

#define STEP_PUSH_BACK(StepClassName) \
        m_vecSteps.push_back(std::shared_ptr<StepClassName>(new StepClassName(pRayCaster , m_pProgram)));

        //Main
        STEP_PUSH_BACK(RCStepMainVert);
        STEP_PUSH_BACK(RCStepMainFrag);

        //Utils
        STEP_PUSH_BACK(RCStepUtils);

        //Composite
        if (m_eCompositeMode == COMPOSITE_DVR)
        {

        }
        else if (m_eCompositeMode == COMPOSITE_AVERAGE)
        {
            STEP_PUSH_BACK(RCStepRayCastingAverage);
            STEP_PUSH_BACK(RCStepCompositeAverage);
        }
        else if (m_eCompositeMode == COMPOSITE_MIP)
        {
            STEP_PUSH_BACK(RCStepRayCastingMIPMinIP);
            STEP_PUSH_BACK(RCStepCompositeMIP);
        }
        else if (m_eCompositeMode == COMPOSITE_MINIP)
        {
            STEP_PUSH_BACK(RCStepRayCastingMIPMinIP);
            STEP_PUSH_BACK(RCStepCompositeMinIP);
        }

        //Mask
        if (m_eMaskMode == MASK_NONE)
        {
            STEP_PUSH_BACK(RCStepMaskNoneSampler);
        }
        else if (m_eMaskMode == MASK_MULTI_LABEL)
        {
            STEP_PUSH_BACK(RCStepMaskNearstSampler);
        }

        //Volume 
        if (m_eInterpolationMode == LINEAR)
        {
            STEP_PUSH_BACK(RCStepVolumeLinearSampler);
        }
        else if (m_eInterpolationMode == NEARST)
        {
            STEP_PUSH_BACK(RCStepVolumeNearstSampler);
        }
        else if (m_eInterpolationMode == CUBIC)
        {
            //TODO
        }

        //Shading
        if (m_eShadingMode == SHADING_NONE)
        {
            STEP_PUSH_BACK(RCStepShadingNone);
        }
        else if (m_eShadingMode == SHADING_PHONG)
        {
            STEP_PUSH_BACK(RCStepShadingPhong);
        }

        //Color inverse
        if (m_eColorInverseMode == COLOR_INVERSE_DISABLE)
        {
            STEP_PUSH_BACK(RCStepColorInverseDisable);
        }
        else //(m_eColorInverseMode == COLOR_INVERSE_ENABLE)
        {
            STEP_PUSH_BACK(RCStepColorInverseEnable);
        }

#undef STEP_PUSH_BACK



        //compile
        std::vector<GLShaderInfo> vecShaders;
        for (auto it = m_vecSteps.begin() ; it != m_vecSteps.end() ; ++it)
        {
            vecShaders.push_back((*it)->get_shader_info());
        }
        m_pProgram->set_shaders(vecShaders);
        m_pProgram->compile();

        for (auto it = m_vecSteps.begin() ; it != m_vecSteps.end() ; ++it)
        {
            (*it)->get_uniform_location();
        }
    }

    CHECK_GL_ERROR;
}

MED_IMAGING_END_NAMESPACE