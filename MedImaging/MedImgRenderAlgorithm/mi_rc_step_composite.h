#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_COMPOSITE_H
#define MED_IMAGING_RAY_CATING_GPU_STEP_COMPOSITE_H

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepCompositeAverage : public RCStepBase
{
public:
    RCStepCompositeAverage(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepCompositeAverage(){};

    virtual GLShaderInfo GetShaderInfo();

private:
};

class RCStepCompositeMIP : public RCStepBase
{
public:
    RCStepCompositeMIP(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepBase(pRayCaster , pProgram)
      {};

      virtual ~RCStepCompositeMIP(){};

      virtual GLShaderInfo GetShaderInfo();

private:
};

class RCStepCompositeMinIP : public RCStepBase
{
public:
    RCStepCompositeMinIP(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepBase(pRayCaster , pProgram),m_iLocCustomMinThreshold(-1)
      {};

      virtual ~RCStepCompositeMinIP(){};

      virtual GLShaderInfo GetShaderInfo();

      virtual void SetGPUParameter();

      virtual void GetUniformLocation();

private:
    int m_iLocCustomMinThreshold;
};

class RCStepCompositeDVR : public RCStepBase
{
public:
    RCStepCompositeDVR(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):
      RCStepBase(pRayCaster , pProgram)
      {};

      virtual ~RCStepCompositeDVR(){};

      virtual GLShaderInfo GetShaderInfo();

      virtual void SetGPUParameter();

      virtual void GetUniformLocation();

private:
};

MED_IMAGING_END_NAMESPACE

#endif