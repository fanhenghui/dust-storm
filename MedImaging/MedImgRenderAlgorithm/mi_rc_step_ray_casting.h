#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_RAY_CASTING_H
#define MED_IMAGING_RAY_CATING_GPU_STEP_RAY_CASTING_H

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE


class RCStepRayCastingMIPBase : public RCStepBase
{
public:
    RCStepRayCastingMIPBase(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepBase(pRayCaster , pProgram),
          m_iLocPseudoColor(-1),
          m_iLocPseudoColorSlope(-1),
          m_iLocPseudoColorIntercept(-1),
          m_iLocGlobalWL(-1)
      {};

      virtual ~RCStepRayCastingMIPBase(){};

      virtual void set_gpu_parameter();

      virtual void get_uniform_location();
    

private:
    int m_iLocPseudoColor;
    int m_iLocPseudoColorSlope;
    int m_iLocPseudoColorIntercept;
    int m_iLocGlobalWL;
};

class RCStepRayCastingAverage : public RCStepRayCastingMIPBase
{
public:
    RCStepRayCastingAverage(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepRayCastingMIPBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepRayCastingAverage(){};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepRayCastingMIPMinIP : public RCStepRayCastingMIPBase
{
public:
    RCStepRayCastingMIPMinIP(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepRayCastingMIPBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepRayCastingMIPMinIP(){};

    virtual GLShaderInfo get_shader_info();

private:;
};

MED_IMAGING_END_NAMESPACE

#endif