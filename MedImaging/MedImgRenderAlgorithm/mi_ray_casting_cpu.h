#ifndef MED_IMAGING_RAY_CASTING_CPU_H_
#define MED_IMAGING_RAY_CASTING_CPU_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgArithmetic/mi_vector4f.h"
#include "MedImgArithmetic/mi_sampler.h"

MED_IMAGING_BEGIN_NAMESPACE

class RayCaster;

class RayCastingCPU
{
public:
    RayCastingCPU(std::shared_ptr<RayCaster> pRayCaster);
    ~RayCastingCPU();
    void render(int iTestCode = 0);
private:
    //For testing
    void render_entry_exit_points_i(int iTestCode);

    //Dispatch render mode
    template<class T>
    void ray_casting_i(std::shared_ptr<RayCaster> pRayCaster);

    //Average
    template<class T>
    void ray_casting_average_i(std::shared_ptr<RayCaster> pRayCaster);

    //MIP
    template<class T>
    void ray_casting_mip_i(std::shared_ptr<RayCaster> pRayCaster);

    //MinIP
    template<class T>
    void ray_casting_minip_i(std::shared_ptr<RayCaster> pRayCaster);

private:
    std::weak_ptr<RayCaster> m_pRayCaster;
    //Cache
    int m_iWidth;
    int m_iHeight;
    Vector4f* m_pEntryPoints;
    Vector4f* m_pExitPoints;
    unsigned int m_uiDim[3];
    void* m_pVolumeDataRaw;
    unsigned char* m_pMaskDataRaw;
    RGBAUnit* m_pColorCanvas;

};



MED_IMAGING_END_NAMESPACE

#endif