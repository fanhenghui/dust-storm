#include "mi_brick_generator.h"
#include "mi_brick_utils.h"
#include "MedImgIO/mi_image_data.h"

#include "boost/thread.hpp"
#include "MedImgCommon/mi_concurrency.h"

MED_IMAGING_BEGIN_NAMESPACE

BrickGenerator::BrickGenerator()
{

}

BrickGenerator::~BrickGenerator()
{

}

 void BrickGenerator::calculate_brick_corner( 
     std::shared_ptr<ImageData> pImgData , 
     unsigned int uiBrickSize , unsigned int uiBrickExpand , 
     BrickCorner* pBrickCorner)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(pBrickCorner);

    unsigned int uiBrickDim[3] = {1,1,1};
    BrickUtils::instance()->get_brick_dim(pImgData->m_uiDim , uiBrickDim , uiBrickSize);

    const unsigned int uiBrickCount = uiBrickDim[0]*uiBrickDim[1]*uiBrickDim[2];

    //BrickCorner* pBrickCorner = new BrickCorner[uiBrickCount];
    unsigned int uiX , uiY , uiZ;
    unsigned int uiLayerCount = uiBrickDim[0]*uiBrickDim[1];
    for (unsigned int i = 0 ; i<uiBrickCount ; ++i)
    {
        uiZ =  i/uiLayerCount;
        uiY = (i - uiZ*uiLayerCount)/uiBrickDim[0];
        uiX = i - uiZ*uiLayerCount - uiY*uiBrickDim[0];
        pBrickCorner[i].m_Min[0] = uiX*uiBrickSize;
        pBrickCorner[i].m_Min[1] = uiY*uiBrickSize;
        pBrickCorner[i].m_Min[2] = uiZ*uiBrickSize;
    }

}

void BrickGenerator::calculate_brick_unit( 
    std::shared_ptr<ImageData> pImgData ,
    BrickCorner* pBrickCorner , 
    unsigned int uiBrickSize , unsigned int uiBrickExpand , 
    BrickUnit* pBrickUnit)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(pBrickUnit);

    unsigned int uiBrickDim[3] = {1,1,1};
    BrickUtils::instance()->get_brick_dim(pImgData->m_uiDim , uiBrickDim , uiBrickSize);

    const unsigned int uiBrickCount = uiBrickDim[0]*uiBrickDim[1]*uiBrickDim[2];
    //BrickUnit* pBrickUnit = new BrickUnit[uiBrickCount];

    const unsigned int uiDispatch = Concurrency::instance()->get_app_concurrency();

    std::vector<boost::thread> vecThreads(uiDispatch-1);
    const unsigned int uiBrickDispath = uiBrickCount/uiDispatch;

    switch(pImgData->m_eDataType)
    {
    case UCHAR:
        {
            if (1 == uiDispatch)
            {
                calculate_brick_unit_kernel_i<unsigned char>(0 , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
                {
                    vecThreads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<unsigned char>, this , 
                        uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand));
                }
                calculate_brick_unit_kernel_i<unsigned char>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
                std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));
            }

            break;
        }
    case USHORT:
        {
            if (1 == uiDispatch)
            {
                calculate_brick_unit_kernel_i<unsigned short>(0 , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
                {
                    vecThreads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<unsigned short>, this , 
                        uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand));
                }
                calculate_brick_unit_kernel_i<unsigned short>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
                std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));
            }
            break;
        }
    case SHORT:
        {
            if (1 == uiDispatch)
            {
                calculate_brick_unit_kernel_i<short>(0 , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
                {
                    vecThreads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<short>, this , 
                        uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand));
                }
                calculate_brick_unit_kernel_i<short>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
                std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));
            }
            break;
        }
    case FLOAT:
        {
            if (1 == uiDispatch)
            {
                calculate_brick_unit_kernel_i<float>(0 , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < uiDispatch - 1 ; ++i)
                {
                    vecThreads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<float>, this , 
                        uiBrickDispath*i , uiBrickDispath*(i+1) , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand));
                }
                calculate_brick_unit_kernel_i<float>(uiBrickDispath*(uiDispatch-1) , uiBrickCount , pBrickCorner , pBrickUnit , pImgData , uiBrickSize , uiBrickExpand);
                std::for_each(vecThreads.begin() , vecThreads.end() , std::mem_fun_ref(&boost::thread::join));
            }
            break;
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Undefined data type!");
        }
    }
}

template<typename T>
void BrickGenerator::calculate_brick_unit_kernel_i( 
    unsigned int uiBegin , unsigned int uiEnd ,
    BrickCorner* pBrickCorner , BrickUnit* pBrickUnit , 
    std::shared_ptr<ImageData> pImgData , 
    unsigned int uiBrickSize , unsigned int uiBrickExpand )
{
    for (unsigned int i = uiBegin ; i < uiEnd ; ++i)
    {
        calculate_brick_unit_i<T>(pBrickCorner[i] , pBrickUnit[i] ,pImgData , uiBrickSize , uiBrickExpand);
    }
}

template<typename T>
void BrickGenerator::calculate_brick_unit_i( 
    BrickCorner& bc , BrickUnit& bu , 
    std::shared_ptr<ImageData> pImgData , 
    unsigned int uiBrickSize , unsigned int uiBrickExpand )
{
    const unsigned int uiBrickLength = uiBrickSize + 2*uiBrickExpand;
    bu.m_pData = new char[sizeof(T)*uiBrickLength*uiBrickLength*uiBrickLength];
    memset(bu.m_pData , 0 , sizeof(T)*(uiBrickLength*uiBrickLength*uiBrickLength));

    T* pDst = (T*)bu.m_pData;
    T* pSrc = (T*)pImgData->get_pixel_pointer();
    unsigned int *uiVolumeDim = pImgData->m_uiDim;

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

    assert(uiEnd[0] - uiBegin[0] == uiEndBrick[0] - uiBeginBrick[0]);
    assert(uiEnd[1] - uiBegin[1] == uiEndBrick[1] - uiBeginBrick[1]);
    assert(uiEnd[2] - uiBegin[2] == uiEndBrick[2] - uiBeginBrick[2]);

    const unsigned int uiVolumeLayerCount = uiVolumeDim[0]*uiVolumeDim[1];
    const unsigned int uiBrickLayerCount = uiBrickLength*uiBrickLength;
    const unsigned int uiCopyLen = uiEndBrick[0] - uiBeginBrick[0];
    const unsigned int uiCopyByte = uiCopyLen*sizeof(T);

    for (unsigned int z = uiBegin[2] , z0 = uiBeginBrick[2] ; z < uiEnd[2] ; ++z , ++z0)
    {
        for (unsigned int y = uiBegin[1] , y0 = uiBeginBrick[1] ; y < uiEnd[1] ; ++y , ++y0)
        {
            memcpy(pDst+z0*uiBrickLayerCount + y0*uiBrickLength + uiBeginBrick[0] , pSrc + z*uiVolumeLayerCount + y*uiVolumeDim[0] + uiBegin[0] , uiCopyByte);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //Test output
    //{
    //    std::stringstream ss;
    //    ss << "D:/temp/brick_data_" << bc.m_Min[0] <<"_"<< bc.m_Min[1] <<"_"<< bc.m_Min[2] <<".raw";
    //    std::ofstream out(ss.str().c_str() , std::ios::out | std::ios::binary);
    //    if (out.is_open())
    //    {
    //        out.write((char*)pDst , sizeof(T)*uiBrickLength*uiBrickLength*uiBrickLength);
    //        out.close();
    //    }
    //}
    //////////////////////////////////////////////////////////////////////////
}



MED_IMAGING_END_NAMESPACE