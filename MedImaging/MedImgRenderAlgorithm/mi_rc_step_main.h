#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_MAIN_H
#define MED_IMAGING_RAY_CATING_GPU_STEP_MAIN_H

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepMainVert : public RCStepBase
{
public:
    RCStepMainVert(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepMainVert(){};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

private:
};

class RCStepMainFrag : public RCStepBase
{
public:
    RCStepMainFrag(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepBase(pRayCaster , pProgram),
          m_iLocVolumeDim(-1),
          m_iLocVolumeData(-1),
          m_iLocMaskData(-1),
          m_iLocSampleRate(-1)
    {

    };

    virtual ~RCStepMainFrag(){};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

    virtual void get_uniform_location();

private:
    int m_iLocVolumeDim;
    int m_iLocVolumeData;
    int m_iLocMaskData;
    int m_iLocSampleRate;
};

MED_IMAGING_END_NAMESPACE

#endif