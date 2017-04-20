#ifndef MED_IMAGING_RAY_CASTING_GPU_H_
#define MED_IMAGING_RAY_CASTING_GPU_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"

MED_IMAGING_BEGIN_NAMESPACE
class RayCaster;
class RCStepBase;
class GLActiveTextureCounter;
class RayCastingGPU
{
public:
    RayCastingGPU(std::shared_ptr<RayCaster> pRayCaster);
    ~RayCastingGPU();
    void render(int iTestCode = 0);
private:
    void update_i();
private:
    std::weak_ptr<RayCaster> m_pRayCaster;
    GLProgramPtr m_pProgram;
    std::shared_ptr<GLActiveTextureCounter> m_pGLActTexCounter;

    //render steps
    std::vector<std::shared_ptr<RCStepBase>> m_vecSteps;

    //Ray casting mode cache
    MaskMode m_eMaskMode;
    CompositeMode m_eCompositeMode;
    InterpolationMode m_eInterpolationMode;
    ShadingMode m_eShadingMode;
    ColorInverseMode m_eColorInverseMode;

    //VAO
    GLVAOPtr m_pVAO;
    GLBufferPtr m_pVertexBuffer;
};

MED_IMAGING_END_NAMESPACE

#endif