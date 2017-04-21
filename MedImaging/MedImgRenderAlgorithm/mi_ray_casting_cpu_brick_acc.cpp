#include "mi_ray_casting_cpu_brick_acc.h"

#include "boost/thread.hpp"

#include "MedImgCommon/mi_concurrency.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgArithmetic/mi_sampler.h"
#include "MedImgArithmetic/mi_ortho_camera.h"
#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"
#include "mi_mpr_entry_exit_points.h"
#include "mi_ray_caster_canvas.h"
#include "mi_brick_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

    namespace//TODO 这个函数要移到碰撞检测算法去
{
    //Return true if out
    bool check_outside(Vector3f pt, Vector3f boundMin , Vector3f boundMax)
    {
        if (pt._m[0] <= boundMin._m[0] || pt._m[1] <= boundMin._m[1] || pt._m[2] < boundMin._m[2]||
            pt._m[0] > boundMax._m[0] || pt._m[1] > boundMax._m[1] || pt._m[2] > boundMax._m[2])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //original
    //bool RayIntersectAABB(Vector3f ptRayStart, Vector3f ptMin, Vector3f vBound, Vector3f vRay, 
    //    float& fEntryStep, float& fExitStep)
    //{
    //    Vector3f vBottom =  (ptMin - ptRayStart);
    //    Vector3f vTop =  (ptMin + vBound - ptRayStart);

    //    Vector3f vBottomStep = vBottom / vRay;
    //    Vector3f vTopStep = vTop / vRay;

    //    Vector3f vMinStep = min_per_elem(vBottomStep, vTopStep);
    //    Vector3f vMaxStep = max_per_elem(vBottomStep, vTopStep);
    //    float fNearStep = vMinStep.max_elem();
    //    float fFarStep = vMaxStep.min_elem();

    //    fEntryStep = std::max(fNearStep, 0.0f);
    //    fExitStep = fFarStep;

    //    return fEntryStep < fExitStep;
    //}
    //////////////////////////////////////////////////////////////////////////

    bool RayIntersectAABB(Vector3f ptRayStart, Vector3f ptMin, Vector3f vBound, Vector3f vRay, 
        float& fEntryStep, float& fExitStep)
    {
        Vector3f vBottomStep =  (ptMin - ptRayStart)/vRay;
        Vector3f vTopStep =  (ptMin + vBound - ptRayStart)/vRay;
        Vector3f vBottomStep2(vBottomStep);
        Vector3f vTopStep2(vTopStep);
        for (int i = 0 ; i< 3 ; ++i)
        {
            if (fabs(vRay._m[i]) <= FLOAT_EPSILON)
            {
                vBottomStep._m[i] = -std::numeric_limits<float>::max();
                vTopStep._m[i] = -std::numeric_limits<float>::max();
                vBottomStep2._m[i] = std::numeric_limits<float>::max();
                vTopStep2._m[i] = std::numeric_limits<float>::max();
            }
        }

        fEntryStep = vBottomStep.min_per_elem(vTopStep).max_elem();
        fExitStep = vBottomStep2.max_per_elem(vTopStep2).min_elem();

        //////////////////////////////////////////////////////////////////////////
        //fNear > fFar not intersected
        //fNear >0  fFar > 0 fNear <= fFar intersected , start point not arrive AABB yet
        //fNear <0 fFar > 0 intersected , start point is in AABB
        //fNear <0 fFar < 0 fNear < fFar , intersected , but start point is outside AABB in extension ray 
        return fEntryStep < fExitStep;
    }

    //If ray[i] < FLOAT_EPSLION then set ray[i] = 1 adjust[i] = std::numeric_limits<float>::max()*0.5f
    bool RayIntersectAABBAcc(Vector3f ptRayStart, Vector3f ptMin, Vector3f vBound, Vector3f vRay, Vector3f vAdjust,
        float& fEntryStep, float& fExitStep)
    {
        Vector3f vBottomStep =  (ptMin - ptRayStart)/vRay;
        Vector3f vTopStep =  (ptMin + vBound - ptRayStart)/vRay;
        Vector3f vBottomStep2(vBottomStep);
        Vector3f vTopStep2(vTopStep);
        vBottomStep -= vAdjust;
        vTopStep -= vAdjust;
        vBottomStep2 += vAdjust;
        vTopStep2 += vAdjust;

        fEntryStep = vBottomStep.min_per_elem(vTopStep).max_elem();
        fExitStep = vBottomStep2.max_per_elem(vTopStep2).min_elem();

        //////////////////////////////////////////////////////////////////////////
        //fNear > fFar not intersected
        //fNear >0  fFar > 0 fNear <= fFar intersected , start point not arrive AABB yet
        //fNear <0 fFar > 0 intersected , start point is in AABB
        //fNear <0 fFar < 0 fNear < fFar , intersected , but start point is outside AABB in extension ray 
        return fEntryStep < fExitStep;
    }
}

bool operator <(const BrickDistance& l , const BrickDistance& r)
{
    return l.m_fDistance < r.m_fDistance;
}

RayCastingCPUBrickAcc::RayCastingCPUBrickAcc(std::shared_ptr<RayCaster> pRayCaster):m_pRayCaster(pRayCaster),
    _width(32),
    _height(32),
    m_pEntryPoints(nullptr),
    m_pExitPoints(nullptr),
    m_pColorCanvas(nullptr),
    m_pBrickCorner(nullptr),
    m_pVolumeBrickUnit(nullptr),
    m_pMaskBrickUnit(nullptr),
    m_pVolumeBrickInfo(nullptr),
    m_pMaskBrickInfo(nullptr),
    m_uiBrickSize(32),
    m_uiBrickExpand(2),
    m_uiBrickCount(0),
    m_uiInterBrickNum(0),
    m_iRayCount(0)
{
    _dim[0] = _dim[1] = _dim[2] = 32;
    m_uiBrickDim[0] = m_uiBrickDim[1] = m_uiBrickDim[2] = 0;

    m_iTestPixelX = 123546;
    m_iTestPixelY = 123546;
}

RayCastingCPUBrickAcc::~RayCastingCPUBrickAcc()
{

}

void RayCastingCPUBrickAcc::render(int iTestCode /*= 0*/)
{
    try
    {
        std::shared_ptr<RayCaster> pRayCaster = m_pRayCaster.lock();
        RENDERALGO_CHECK_NULL_EXCEPTION(pRayCaster);

        //Volume info
        RENDERALGO_CHECK_NULL_EXCEPTION(pRayCaster->m_pEntryExitPoints);
        pRayCaster->m_pEntryExitPoints->get_display_size(_width , _height);

        std::shared_ptr<ImageData> pVolumeData = pRayCaster->m_pVolumeData;
        RENDERALGO_CHECK_NULL_EXCEPTION(pVolumeData);
        memcpy(_dim , pVolumeData->_dim , sizeof(unsigned int)*3);

        //Brick struct
        m_uiBrickSize = pRayCaster->m_uiBrickSize;
        m_uiBrickExpand = pRayCaster->m_uiBrickExpand;
        m_pBrickCorner = pRayCaster->m_pBrickCorner;
        m_pVolumeBrickUnit = pRayCaster->m_pVolumeBrickUnit;
        m_pMaskBrickUnit = pRayCaster->m_pMaskBrickUnit;
        m_pVolumeBrickInfo = pRayCaster->m_pVolumeBrickInfo;
        m_pMaskBrickInfo = pRayCaster->m_pMaskBrickInfo;
        unsigned int uiBrickDim[3] = {1,1,1};
        BrickUtils::instance()->get_brick_dim(_dim , uiBrickDim , m_uiBrickSize);
        m_uiBrickCount = uiBrickDim[0]*uiBrickDim[1]*uiBrickDim[2];
        if( !(m_uiBrickDim[0] ==uiBrickDim[0] && m_uiBrickDim[1] == uiBrickDim[1] && m_uiBrickDim[2] == uiBrickDim[2]) )
        {
            memcpy(m_uiBrickDim , uiBrickDim , sizeof(unsigned int)*3);
            m_vecBrickCenterDistance.clear();
            m_vecBrickCenterDistance.resize(m_uiBrickCount);
        }

        if (m_iRayCount != _width*_height)
        {
            m_iRayCount = _width*_height;
            m_pRayResult.reset(new float[m_iRayCount]);
        }
        //memset(m_pRayResult.get() , 0 , sizeof(float)*m_iRayCount);


        //Entry exit points
        m_pEntryPoints = pRayCaster->m_pEntryExitPoints->get_entry_points_array();
        m_pExitPoints = pRayCaster->m_pEntryExitPoints->get_exit_points_array();

        //Canvas
        RENDERALGO_CHECK_NULL_EXCEPTION(pRayCaster->m_pCanvas);
        m_pColorCanvas = pRayCaster->m_pCanvas->get_color_array();
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pColorCanvas);
        memset(m_pColorCanvas , 0 , sizeof(RGBAUnit)*m_iRayCount);
        //if ()
        {
            //TODO memset gray
        }

        //Matrix
        const Matrix4 matV2W = pRayCaster->m_matVolume2World;
        const Matrix4 matVP = pRayCaster->m_pCamera->get_view_projection_matrix();
        const Matrix4 matMVP = matVP*matV2W;
        const Matrix4 matMVPInv = matMVP.get_inverse();
        m_matMVP = ArithmeticUtils::convert_matrix(matMVP);
        m_matMVPInv = ArithmeticUtils::convert_matrix(matMVPInv);
        m_matMVPInv0 = matMVPInv;

        //////////////////////////////////////////////////////////////////////////
        //1 Brick sort
        clock_t t0 = clock();
        sort_brick_i();

        clock_t t1= clock();
        std::cout << "Sort brick cost : " << double(t1 - t0) << " ms.\n";

        //2 Brick ray casting
        for (unsigned int i = 0 ; i<m_uiInterBrickNum ; ++i)
        {
            ray_casting_in_brick_i(m_vecBrickCenterDistance[i].m_id , pRayCaster);
        }
        clock_t t2= clock();
        std::cout << "Ray casting cost : " << double(t2 - t1) << " ms.\n";


        //////////////////////////////////////////////////////////////////////////
        //3 update color to texture
        pRayCaster->m_pCanvas->update_color_array();

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
}

void RayCastingCPUBrickAcc::sort_brick_i()
{
    std::shared_ptr<RayCaster> pRayCaster = m_pRayCaster.lock();
    std::shared_ptr<MPREntryExitPoints> pMPREntryExitPoints = std::dynamic_pointer_cast<MPREntryExitPoints>(pRayCaster->m_pEntryExitPoints);
    if (pMPREntryExitPoints)//MPR
    {
        //1 Get bricks between entry and exit
        const Matrix4 matW2V = pRayCaster->m_matVolume2World.get_inverse();
        const Point3 ptEyeD = matW2V.transform(pRayCaster->m_pCamera->get_eye());
        const Vector3f ptEye((float)ptEyeD.x , (float)ptEyeD.y ,(float)ptEyeD.z);
        Vector3f ptCenter;
        const float fBrickSize = (float)m_uiBrickSize;
        Vector3f vHalfBrickBound(fBrickSize*0.5f);

        Vector4f vEntryPlane;
        Vector4f vExitPlane;
        pMPREntryExitPoints->get_entry_exit_plane(vEntryPlane , vExitPlane, m_vRayDirNorm);
        Vector4f ptMin , ptMax ;
        Vector4f pt[8];
        const Vector4f vBrickBound(fBrickSize ,fBrickSize ,fBrickSize , 0);
        float fMinEntry(0.0f) , fMaxEntry(0.0f) , fMinExit(0.0f) , fMaxExit(0.0f);
        m_uiInterBrickNum = 0;
        for (unsigned int i = 0 ; i < m_uiBrickCount ; ++i)
        {
            const BrickCorner &bc = m_pBrickCorner[i];
            ptMin = Vector4f((float)bc.m_Min[0] , (float)bc.m_Min[1] , (float)bc.m_Min[2] , -1.0f);
            ptMax = ptMin + vBrickBound;

            pt[0] = ptMin;
            pt[1] = Vector4f(ptMin._m[0],ptMin._m[1],ptMax._m[2], -1.0f);
            pt[1] =Vector4f(ptMin._m[0],ptMin._m[1],ptMax._m[2], -1.0f);
            pt[2] =Vector4f(ptMin._m[0],ptMax._m[1],ptMin._m[2], -1.0f);
            pt[3] =Vector4f(ptMin._m[0],ptMax._m[1],ptMax._m[2], -1.0f);
            pt[4] =Vector4f(ptMax._m[0],ptMin._m[1],ptMin._m[2], -1.0f);
            pt[5] =Vector4f(ptMax._m[0],ptMin._m[1],ptMax._m[2], -1.0f);
            pt[6] =Vector4f(ptMax._m[0],ptMax._m[1],ptMin._m[2], -1.0f);
            pt[7] = ptMax;

            //Intersect with entry plane
            int iEntryPos = 0;
            int iEntryNeg = 0;
            for ( int j = 0 ; j<8 ; ++j)
            {
                if (pt[j].dot_product(vEntryPlane) > 0.0f)
                {
                    iEntryPos += 1;
                }
                else
                {
                    iEntryNeg +=1;
                }
            }

            if (8 != iEntryPos && 8 != iEntryNeg)//intersection
            {
                ptCenter = Vector3f(ptMin.m_Vec128) + vHalfBrickBound;

                m_vecBrickCenterDistance[m_uiInterBrickNum].m_id = i;
                m_vecBrickCenterDistance[m_uiInterBrickNum].m_fDistance = (ptCenter - ptEye).dot_product(vEntryPlane.get_128());
                ++m_uiInterBrickNum;
                continue;
            }

            int iExitPos = 0;
            int iExitNeg = 0;
            for ( int j = 0 ; j<8 ; ++j)
            {
                if (pt[j].dot_product(vExitPlane) > 0.0f)
                {
                    iExitPos += 1;
                }
                else
                {
                    iExitNeg +=1;
                }
            }

            if (8 != iExitPos && 8 != iExitNeg)//intersection
            {
                ptCenter = Vector3f(ptMin.m_Vec128) + vHalfBrickBound;

                m_vecBrickCenterDistance[m_uiInterBrickNum].m_id = i;
                m_vecBrickCenterDistance[m_uiInterBrickNum].m_fDistance = (ptCenter - ptEye).dot_product(vEntryPlane.get_128());
                ++m_uiInterBrickNum;
                continue;
            }

            if (8 == iEntryPos && 8 == iExitPos)
            {
                ptCenter = Vector3f(ptMin.m_Vec128) + vHalfBrickBound;

                m_vecBrickCenterDistance[m_uiInterBrickNum].m_id = i;
                m_vecBrickCenterDistance[m_uiInterBrickNum].m_fDistance = (ptCenter - ptEye).dot_product(vEntryPlane.get_128());
                ++m_uiInterBrickNum;
                continue;
            }
        }

        std::cout << "Intersect brick count : " << m_uiInterBrickNum << std::endl;

        //2 Sort bricks
        if (m_uiInterBrickNum > 1)
        {
            std::sort(m_vecBrickCenterDistance.begin() , m_vecBrickCenterDistance.begin() + m_uiInterBrickNum , std::less<BrickDistance>());
        }

    }
    else//Maybe VR
    {
        RENDERALGO_THROW_EXCEPTION("Entry exit points is not MPR!");
    }
}
const std::vector<BrickDistance>& RayCastingCPUBrickAcc::get_brick_distance() const
{
    return m_vecBrickCenterDistance;
}

unsigned int RayCastingCPUBrickAcc::get_ray_casting_brick_count() const
{
    return m_uiInterBrickNum;
}

void RayCastingCPUBrickAcc::ray_casting_in_brick_i(unsigned int uiBrickID ,  const std::shared_ptr<RayCaster>& pRayCaster)
{
    //1 inverse projection
    const BrickCorner &bc = m_pBrickCorner[uiBrickID];
    const Vector3f vBrickBound((float)m_uiBrickSize);
    //const Vector3f vBrickExpand((float)m_uiBrickExpand);
    const Vector3f ptMin = Vector3f((float)bc.m_Min[0] , (float)bc.m_Min[1] , (float)bc.m_Min[2]);
    const Vector3f ptMax = ptMin + vBrickBound;
    //const unsigned int uiBrickSampleSize = m_uiBrickExpand + m_uiBrickSize;

    const Vector3f pt[8] = {ptMin , 
        Vector3f(ptMin._m[0],ptMin._m[1],ptMax._m[2]),
        Vector3f(ptMin._m[0],ptMax._m[1],ptMin._m[2]),
        Vector3f(ptMin._m[0],ptMax._m[1],ptMax._m[2]),
        Vector3f(ptMax._m[0],ptMin._m[1],ptMin._m[2]),
        Vector3f(ptMax._m[0],ptMin._m[1],ptMax._m[2]),
        Vector3f(ptMax._m[0],ptMax._m[1],ptMin._m[2]),
        ptMax};

    const float fWidth = (float)_width;
    const float fHeight = (float)_height;
    int iXBegin(65535), iXEnd(-65535) , iYBegin(65535) , iYEnd(-65535);
    int iCurX , iCurY;
    float fNormX , fNormY;
    Vector3f vScreen;
    int iOut = 1;//completely outside of screen
    int iBrickVertexOut = 0;
    for (int i = 0 ; i< 8 ; ++i)
    {
        vScreen = m_matMVP.transform_point(pt[i]);
        fNormX = vScreen._m[0];
        fNormY = vScreen._m[1];

        iOut = 1;
        if (fNormX < -1)
        {
            fNormX = -1;
            iOut+=1;
        }
        else if (fNormX > 1)
        {
            fNormX = 1;
            iOut+=1;
        }

        if (fNormY < -1)
        {
            fNormY = -1;
            iOut+=1;
        }
        else if (fNormY > 1)
        {
            fNormY = 1;
            iOut+=1;
        }

        iBrickVertexOut += (iOut>>1);


        //if (fNormX < -1 || fNormX > 1||
        //    fNormY < -1 || fNormY > 1)
        //{
        //    iBrickVertexOut +=1;
        //    continue;
        //}

        iCurX = int( (fNormX+1.0f)*0.5f * fWidth + 0.5);
        if (iCurX < iXBegin)
        {
            iXBegin = iCurX;
        }
        if (iCurX > iXEnd)
        {
            iXEnd = iCurX;
        }

        iCurY = int( (fNormY+1.0f)*0.5f*fHeight + 0.5);
        if (iCurY < iYBegin)
        {
            iYBegin = iCurY;
        }
        if (iCurY > iYEnd)
        {
            iYEnd = iCurY;
        }
    }

    if (iYEnd > _height || iXEnd > _width)
    {
        //std::cout <<"ERR\n";
    }

    if (iBrickVertexOut < 8)
    {
        //Multi-thread
        const int iXRange = iXEnd - iXBegin;
        const int iYRange = iYEnd - iYBegin;
        const int iPixelNum = iXRange*iYRange;
        int iQuadRange[4] = {iXBegin ,iXRange , iYBegin , iYRange};


        const Vector3f vBrickBound((float)m_uiBrickSize);
        const Vector3f vBrickExpand((float)m_uiBrickExpand);
        const unsigned int uiBrickSampleSize = m_uiBrickSize + m_uiBrickExpand*2;
        const Vector3f vRayDirSample(m_vRayDirNorm*pRayCaster->m_fSampleRate);
        Sampler<unsigned short> sampler;
        unsigned short* pData = (unsigned short*)m_pVolumeBrickUnit[uiBrickID].m_pData;

        //////////////////////////////////////////////////////////////////////////
        //Adjust ray direction
        Vector3f vRayBrick(vRayDirSample);
        Vector3f vRayBrickAdjust(0,0,0);
        for (int i = 0 ; i< 3 ; ++i)
        {
            if (fabs(vRayBrick._m[i]) <= FLOAT_EPSILON)
            {
                vRayBrick._m[i] = 1;//be divided
                vRayBrickAdjust._m[i] = std::numeric_limits<float>::max()*0.5f;
            }
        }
        //////////////////////////////////////////////////////////////////////////

        const float fMinGray = pRayCaster->m_fGlobalWL - pRayCaster->m_fGlobalWW*0.5f;
        const float fWWR = 1.0f/pRayCaster->m_fGlobalWW;

#pragma omp parallel for 
        for (int iPixelID = 0; iPixelID < iPixelNum ; ++iPixelID)
        {
            int y = iPixelID / iXRange;
            int x = iPixelID - y*iXRange;
            x += iXBegin;
            y += iYBegin;
            const int iRayID = y*_width + x;
            if (x == m_iTestPixelX && y == m_iTestPixelY)
            {
                std::cout <<"ERR";
            }

            float fLastStep , fEndStep , fCurBrickStep;
            Vector3f ptStart;
            float fEntryStep , fExitStep;
            Vector3f vSamplePos;
            float fMaxGray = -65535.0f;
            float fSampleValue , fGray;

            Vector4f &ptEntryPoints = m_pEntryPoints[iRayID];//Use entry points array to store ray parameter
            fLastStep = ptEntryPoints._m[3];
            fEndStep = m_pExitPoints[iRayID]._m[3];

            if (fLastStep >= fEndStep) //brick ray casting end
            {
                continue;
            }

            if (fLastStep < -0.5f) //Skip points which decided in entry points calculation
            {
                m_pColorCanvas[iRayID] =RGBAUnit();
                //const bool bIntersection = RayIntersectAABB(ptEntryF, Vector3f(0,0,0), vDim, vRayDir, fEntryStep, fExitStep);
            }
            else
            {

                //1 Get entry exit points
                ptStart.m_Vec128 = (m_pEntryPoints[iRayID].m_Vec128);

                if (fLastStep < FLOAT_EPSILON)//Zero step, the first step
                {
                    m_pRayResult[iRayID] = -65535.0f;//initialize max gray
                }
                else
                {
                    ptStart += vRayDirSample*(fLastStep + 1.0f);//Current point , step forward once
                }

                if(check_outside(ptStart , ptMin , ptMin+vBrickBound))
                {
                    continue;
                }

                if (RayIntersectAABBAcc(ptStart, ptMin, vBrickBound, vRayBrick, vRayBrickAdjust , fEntryStep, fExitStep))
                {
                    fMaxGray = -65535.0f;
                    fSampleValue = 0.0f;

                    fCurBrickStep = (float)(int(fExitStep - fEntryStep + 0.5f));
                    if (fCurBrickStep + fLastStep > fEndStep)
                    {
                        fCurBrickStep = fEndStep - fLastStep;
                    }

                    for (float fSampleStep = 0.0f ; fSampleStep < fCurBrickStep-0.001f ; fSampleStep+=1.0f)
                    {
                        vSamplePos = ptStart + vRayDirSample*fSampleStep;
                        vSamplePos = vSamplePos - ptMin + vBrickExpand; 
                        fSampleValue = sampler.sample_3d_linear(vSamplePos._m[0] , vSamplePos._m[1] , vSamplePos._m[2] , 
                            uiBrickSampleSize , uiBrickSampleSize , uiBrickSampleSize , pData);
                        fMaxGray = fSampleValue > fMaxGray ?  fSampleValue : fMaxGray;
                    }

                    if (fMaxGray > m_pRayResult[iRayID])
                    {
                        m_pRayResult[iRayID] = fMaxGray;
                        fGray = (fMaxGray - fMinGray)*fWWR;
                        m_pColorCanvas[iRayID] = RGBAUnit(fGray , fGray , fGray);
                    }

                    ptEntryPoints._m[3] = fLastStep + fCurBrickStep;
                }
            }
        }

    }
}




MED_IMAGING_END_NAMESPACE