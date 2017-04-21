#include "mi_brick_info_generator.h"

#include "boost/thread.hpp"

#include "MedImgCommon/mi_concurrency.h"
#include "MedImgIO/mi_image_data.h"

#include "mi_brick_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

CPUVolumeBrickInfoGenerator::CPUVolumeBrickInfoGenerator()
{

}

CPUVolumeBrickInfoGenerator::~CPUVolumeBrickInfoGenerator()
{

}

void CPUVolumeBrickInfoGenerator::calculate_brick_info( 
std::shared_ptr<ImageData> image_data , 
unsigned int uiBrickSize , 
unsigned int uiBrickExpand , 
BrickCorner* pBrickCorner , 
BrickUnit* pBrickUnit , 
VolumeBrickInfo* pBrickInfo )
{
    RENDERALGO_CHECK_NULL_EXCEPTION(pBrickUnit);

    unsigned int uiBrickDim[3] = {1,1,1};
    BrickUtils::instance()->get_brick_dim(image_data->_dim , uiBrickDim , uiBrickSize);

    const unsigned int uiBrickCount = uiBrickDim[0]*uiBrickDim[1]*uiBrickDim[2];

    const unsigned int uiDispatch = Concurrency::instance()->get_app_concurrency();

    std::vector<boost::thread> vecThreads(uiDispatch-1);
    const unsigned int uiBrickDispath = uiBrickCount/uiDispatch;

    switch(image_data->_data_type)
    {
    case UCHAR:
        {
            for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
            {
                vecThreads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<unsigned char>, this , 
                    uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit , pBrickInfo , image_data , uiBrickSize , uiBrickExpand));
            }
            calculate_brick_info_kernel_i<unsigned char>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit ,  pBrickInfo , image_data , uiBrickSize , uiBrickExpand);
            std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    case USHORT:
        {
            for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
            {
                vecThreads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<unsigned short>, this , 
                    uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit ,  pBrickInfo , image_data , uiBrickSize , uiBrickExpand));
            }
            calculate_brick_info_kernel_i<unsigned short>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit ,  pBrickInfo , image_data , uiBrickSize , uiBrickExpand);
            std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    case SHORT:
        {
            for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
            {
                vecThreads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<short>, this , 
                    uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit ,  pBrickInfo , image_data , uiBrickSize , uiBrickExpand));
            }
            calculate_brick_info_kernel_i<short>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit ,  pBrickInfo , image_data , uiBrickSize , uiBrickExpand);
            std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    case FLOAT:
        {
            for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
            {
                vecThreads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<float>, this , 
                    uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit ,  pBrickInfo , image_data , uiBrickSize , uiBrickExpand));
            }
            calculate_brick_info_kernel_i<float>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit ,  pBrickInfo , image_data , uiBrickSize , uiBrickExpand);
            std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Undefined data type!");
        }
    }
}


template<typename T>
void CPUVolumeBrickInfoGenerator::calculate_brick_info_i( 
    BrickCorner& bc , 
    BrickUnit& bu , 
    VolumeBrickInfo& vbi,
    std::shared_ptr<ImageData> image_data , 
    unsigned int uiBrickSize , 
    unsigned int uiBrickExpand )
{
    unsigned int *uiVolumeDim = image_data->_dim;

    const unsigned int uiBrickLength = uiBrickSize + 2*uiBrickExpand;

    unsigned int uiBegin[3] = {bc.m_Min[0] , bc.m_Min[1], bc.m_Min[2]};
    unsigned int uiEnd[3] = {bc.m_Min[0]  + uiBrickSize + uiBrickExpand, bc.m_Min[1] + uiBrickSize + uiBrickExpand, bc.m_Min[2] + uiBrickSize + uiBrickExpand};

    unsigned int uiBeginBrick[3] = {0,0,0};
    unsigned int uiEndBrick[3] = {uiBrickLength,uiBrickLength,uiBrickLength};

    for ( int i= 0 ; i < 3 ; ++i)
    {
        if (uiBegin[i] < uiBrickExpand)
        {
            uiBeginBrick[i] += uiBrickExpand - uiBegin[i];
            uiBegin[i] = 0;
        }
        else
        {
            uiBegin[i] -= uiBrickExpand;
        }

        if (uiEnd[i] > uiVolumeDim[i])
        {
            uiEndBrick[i] -= (uiEnd[i] - uiVolumeDim[i]);
            uiEnd[i] = uiVolumeDim[i];
        }
    }

    float fMax = -65535.0f;
    float fMin = 65535.0f;
    T* pBrickData = (T*)bu.m_pData;
    T curValue = 0;
    float fCurValue = 0;
    const unsigned int uiBrickLayerCount = uiBrickLength*uiBrickLength;
    for (unsigned int z = uiBeginBrick[2] ; z < uiEndBrick[2] ; ++z)
    {
        for (unsigned int y = uiBeginBrick[1] ; y < uiEndBrick[1] ; ++y)
        {
            for (unsigned int x = uiBeginBrick[0] ; x < uiEndBrick[0] ; ++x)
            {
                curValue = pBrickData[z*uiBrickLayerCount + y*uiBrickLength + x];
                fCurValue = (float)curValue;
                fMax = fCurValue > fMax ? fCurValue : fMax;
                fMin = fCurValue < fMin ? fCurValue : fMin;
            }
        }
    }

    vbi.m_fMin = fMin;
    vbi.m_fMax = fMax;
}


template<typename T>
void CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i( 
    unsigned int uiBegin , 
    unsigned int uiEnd , 
    BrickCorner* pBrickCorner , 
    BrickUnit* pBrickUnit , 
    VolumeBrickInfo* pBrickInfo,
    std::shared_ptr<ImageData> image_data , 
    unsigned int uiBrickSize , 
    unsigned int uiBrickExpand )
{
    for (unsigned int i = uiBegin ; i < uiEnd ; ++i)
    {
        calculate_brick_info_i<T>(pBrickCorner[i] , pBrickUnit[i] ,pBrickInfo[i] , image_data , uiBrickSize , uiBrickExpand);
    }
}



MED_IMAGING_END_NAMESPACE