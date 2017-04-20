#include "mi_ray_casting_cpu.h"

#include "boost/thread.hpp"

#include "MedImgIO/mi_image_data.h"
#include "MedImgCommon/mi_concurrency.h"
#include "MedImgArithmetic/mi_sampler.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"
#include "mi_ray_caster_canvas.h"

MED_IMAGING_BEGIN_NAMESPACE

    RayCastingCPU::RayCastingCPU(std::shared_ptr<RayCaster> pRayCaster):m_pRayCaster(pRayCaster),
    m_iWidth(32),
    m_iHeight(32),
    m_pEntryPoints(nullptr),
    m_pExitPoints(nullptr),
    m_pVolumeDataRaw(nullptr),
    m_pMaskDataRaw(nullptr),
    m_pColorCanvas(nullptr)
{
    m_uiDim[0] = m_uiDim[1] = m_uiDim[2] = 32;
}

RayCastingCPU::~RayCastingCPU()
{

}

void RayCastingCPU::render(int iTestCode )
{
    try
    {
        std::shared_ptr<RayCaster> pRayCaster = m_pRayCaster.lock();
        RENDERALGO_CHECK_NULL_EXCEPTION(pRayCaster);

        //Volume info
        RENDERALGO_CHECK_NULL_EXCEPTION(pRayCaster->m_pEntryExitPoints);
        pRayCaster->m_pEntryExitPoints->get_display_size(m_iWidth , m_iHeight);

        std::shared_ptr<ImageData> pVolumeData = pRayCaster->m_pVolumeData;
        RENDERALGO_CHECK_NULL_EXCEPTION(pVolumeData);
        memcpy(m_uiDim , pVolumeData->m_uiDim , sizeof(unsigned int)*3);
        m_pVolumeDataRaw = pVolumeData->get_pixel_pointer();

        //Entry exit points
        m_pEntryPoints = pRayCaster->m_pEntryExitPoints->get_entry_points_array();
        m_pExitPoints = pRayCaster->m_pEntryExitPoints->get_exit_points_array();

        //Canvas
        RENDERALGO_CHECK_NULL_EXCEPTION(pRayCaster->m_pCanvas);
        m_pColorCanvas = pRayCaster->m_pCanvas->get_color_array();
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pColorCanvas);

        //////////////////////////////////////////////////////////////////////////
        //For testing entry & exit points
        if (0 != iTestCode)
        {
           render_entry_exit_points_i(iTestCode);
            pRayCaster->m_pCanvas->update_color_array();
            return;
        }
        //////////////////////////////////////////////////////////////////////////



        switch(pVolumeData->m_eDataType)
        {
        case USHORT:
            {
                ray_casting_i<unsigned short>( pRayCaster );
                break;
            }

        case SHORT:
            {
                ray_casting_i<short>( pRayCaster );
                break;
            }

        case FLOAT:
            {
                ray_casting_i<float>( pRayCaster );
                break;
            }
        default:
            RENDERALGO_THROW_EXCEPTION("Undefined data type!");
        }

        CHECK_GL_ERROR;

        if (COMPOSITE_AVERAGE == pRayCaster->m_eCompositeMode ||
            COMPOSITE_MIP == pRayCaster->m_eCompositeMode ||
            COMPOSITE_MINIP == pRayCaster->m_eCompositeMode)
        {
            pRayCaster->m_pCanvas->update_color_array();
        }

        CHECK_GL_ERROR;

    }
    catch (const Exception& e)
    {
#ifdef _DEBUG
        //TODO LOG
        std::cout << e.what();
#endif
        assert(false);
        throw e;
    }
    catch (const std::exception& e)
    {
#ifdef _DEBUG
        //TODO LOG
        std::cout << e.what();
#endif
        assert(false);
        throw e;
    }
}


template<class T>
void RayCastingCPU::ray_casting_i(std::shared_ptr<RayCaster> pRayCaster )
{
    switch(pRayCaster->m_eCompositeMode)
    {
    case COMPOSITE_AVERAGE:
        {
            ray_casting_average_i<T>(pRayCaster);
            break;
        }
    case COMPOSITE_MIP:
        {
            ray_casting_mip_i<T>(pRayCaster );
            break;
        }
    case COMPOSITE_MINIP:
        {
            ray_casting_minip_i<T>(pRayCaster);
            break;
        }
    default:
        break;
    }
}

template<class T>
void RayCastingCPU::ray_casting_average_i(std::shared_ptr<RayCaster> pRayCaster)
{
    const Sampler<T> sampler;
    const int iTotalPixelNum = m_iWidth*m_iHeight;

#ifndef _DEBUG
#pragma omp parallel for
#endif
    for (int idx = 0; idx<iTotalPixelNum  ; ++idx)
    {
        const int iY = idx / m_iWidth;
        const int iX = idx - iY*m_iWidth;

        //1 Get entry exit points
        const Vector3f ptStart(m_pEntryPoints[idx].m_Vec128);
        const Vector3f ptEnd(m_pExitPoints[idx].m_Vec128);

        const bool bSkip = ptStart._m[3] < -0.5f; // -1.0 for skip , 0  for valid entry exit points
        if (bSkip)
        {
            m_pColorCanvas[idx] = RGBAUnit();
            continue;
        }

        const Vector3f vDir = ptEnd - ptStart;
        const float fLength = vDir.magnitude();
        const Vector3f vDirStep = vDir.get_normalize()*pRayCaster->m_fSampleRate;
        const float fStep = fLength / pRayCaster->m_fSampleRate;
        int iStep =(int)fStep;
        if (iStep == 0)//保证至少积分一次
        {
            iStep = 1;
        }

        //2 Integrate
        const float fRatio =1000.0f;
        const float fRatioR = 1.0f/1000.0f;
        float fSum = 0.0f;
        Vector3f ptSamplePos = ptStart;

        float fSampleValue = 0.0f;
        for (int i = 0 ; i < iStep ; ++i)
        {
            ptSamplePos += ( vDirStep * float(i) );

            fSampleValue = sampler.sample_3d_linear(
                ptSamplePos._m[0] , ptSamplePos._m[1] , ptSamplePos._m[2] , 
                m_uiDim[0], m_uiDim[1], m_uiDim[2],
                (T*)m_pVolumeDataRaw);

            fSum += fSampleValue*fRatioR;
        }
        const float fResult  = fSum *(1.0f/iStep) * fRatio;

        //3Apply window level
        const float fMinGray = pRayCaster->m_fGlobalWL - pRayCaster->m_fGlobalWW*0.5f;
        const float fGray = (fResult - fMinGray)/pRayCaster->m_fGlobalWW;

        //4Apply pseudo color
        //TODO just gray
        m_pColorCanvas[idx] = RGBAUnit(fGray , fGray , fGray);
    }

}

template<class T>
void RayCastingCPU::ray_casting_mip_i( std::shared_ptr<RayCaster> pRayCaster)
{
    const Sampler<T> sampler;
    const int iTotalPixelNum = m_iWidth*m_iHeight;

#pragma omp parallel for
    for (int idx = 0; idx<iTotalPixelNum  ; ++idx)
    {
        const int iY = idx / m_iWidth;
        const int iX = idx - iY*m_iWidth;

        //1 Get entry exit points
        const Vector3f ptStart(m_pEntryPoints[idx].m_Vec128);
        const Vector3f ptEnd(m_pExitPoints[idx].m_Vec128);

        const bool bSkip = ptStart._m[3] <  -0.5f; // -1.0 for skip , 0  for valid entry exit points
        if (bSkip)
        {
            m_pColorCanvas[idx] = RGBAUnit();
            continue;
        }

        const Vector3f vDir = ptEnd - ptStart;
        const float fLength = vDir.magnitude();
        const Vector3f vDirStep = vDir.get_normalize()*pRayCaster->m_fSampleRate;
        const float fStep = fLength / pRayCaster->m_fSampleRate;
        int iStep =(int)fStep;
        if (iStep == 0)//保证至少积分一次
        {
            iStep = 1;
        }

        //2 Integrate
        float fMaxGray = -65535.0f;
        Vector3f ptSamplePos = ptStart;

        float fSampleValue = 0.0f;
        for (int i = 0 ; i < iStep ; ++i)
        {
            ptSamplePos += ( vDirStep * float(i) );

            fSampleValue = sampler.sample_3d_linear(
                ptSamplePos._m[0] , ptSamplePos._m[1] , ptSamplePos._m[2] , 
                m_uiDim[0], m_uiDim[1], m_uiDim[2],
                (T*)m_pVolumeDataRaw);

            fMaxGray = fSampleValue > fMaxGray ? fSampleValue : fMaxGray;
        }

        //3Apply window level
        const float fMinGray = pRayCaster->m_fGlobalWL - pRayCaster->m_fGlobalWW*0.5f;
        const float fGray = (fMaxGray - fMinGray)/pRayCaster->m_fGlobalWW;

        //4Apply pseudo color
        //TODO just gray
        m_pColorCanvas[idx] = RGBAUnit(fGray , fGray , fGray);
    }
}

template<class T>
void RayCastingCPU::ray_casting_minip_i( std::shared_ptr<RayCaster> pRayCaster)
{
    const Sampler<T> sampler;
    const int iTotalPixelNum = m_iWidth*m_iHeight;

#pragma omp parallel for
    for (int idx = 0; idx<iTotalPixelNum  ; ++idx)
    {
        const int iY = idx / m_iWidth;
        const int iX = idx - iY*m_iWidth;

        //1 Get entry exit points
        const Vector3f ptStart(m_pEntryPoints[idx].m_Vec128);
        const Vector3f ptEnd(m_pExitPoints[idx].m_Vec128);

        const bool bSkip = ptStart._m[3] < -0.5f; // -1.0 for skip , 0  for valid entry exit points
        if (bSkip)
        {
            m_pColorCanvas[idx] = RGBAUnit();
            continue;
        }

        const Vector3f vDir = ptEnd - ptStart;
        const float fLength = vDir.magnitude();
        const Vector3f vDirStep = vDir.get_normalize()*pRayCaster->m_fSampleRate;
        const float fStep = fLength / pRayCaster->m_fSampleRate;
        int iStep =(int)fStep;
        if (iStep == 0)//保证至少积分一次
        {
            iStep = 1;
        }

        //2 Integrate
        float fMaxGray = -65535.0f;
        Vector3f ptSamplePos = ptStart;

        float fSampleValue = 0.0f;
        for (int i = 0 ; i < iStep ; ++i)
        {
            ptSamplePos += ( vDirStep * float(i) );

            fSampleValue = sampler.sample_3d_linear(
                ptSamplePos._m[0] , ptSamplePos._m[1] , ptSamplePos._m[2] , 
                m_uiDim[0], m_uiDim[1], m_uiDim[2],
                (T*)m_pVolumeDataRaw);

            fMaxGray = fSampleValue > fMaxGray ? fSampleValue : fMaxGray;
        }

        //3Apply window level
        const float fMinGray = pRayCaster->m_fGlobalWL - pRayCaster->m_fGlobalWW*0.5f;
        const float fGray = (fMaxGray - fMinGray)/pRayCaster->m_fGlobalWW;

        //4Apply pseudo color
        //TODO just gray
        m_pColorCanvas[idx] = RGBAUnit(fGray , fGray , fGray);
    }
}

void RayCastingCPU::render_entry_exit_points_i( int iTestCode)
{
    Vector3f vDimR(1.0f/m_uiDim[0] , 1.0f/m_uiDim[1] , 1.0f/m_uiDim[2]);
    const int iTotalPixelNum = m_iWidth*m_iHeight;
    if (1 == iTestCode)
    {
        for (int i = 0 ; i < iTotalPixelNum ; ++i)
        {
            Vector3f ptStart(m_pEntryPoints[i].m_Vec128);
            ptStart =ptStart*vDimR;
            m_pColorCanvas[i] = RGBAUnit(ptStart._m[0] , ptStart._m[1] , ptStart._m[2]);
        }
    }
    else
    {
        for (int i = 0 ; i < iTotalPixelNum ; ++i)
        {
            Vector3f ptEnd(m_pExitPoints[i].m_Vec128);
            ptEnd =ptEnd*vDimR;
            m_pColorCanvas[i] = RGBAUnit(ptEnd._m[0] , ptEnd._m[1] , ptEnd._m[2]);
        }
    }


}


MED_IMAGING_END_NAMESPACE