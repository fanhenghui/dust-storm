#include "mi_brick_pool.h"

#include "MedImgCommon/mi_configuration.h"
#include "MedImgCommon/mi_concurrency.h"

#include "MedImgIO/mi_image_data.h"

#include "mi_brick_utils.h"
#include "mi_brick_generator.h"
#include "mi_brick_info_generator.h"

MED_IMAGING_BEGIN_NAMESPACE

BrickPool::BrickPool()
{
    m_uiBrickSize = BrickUtils::instance()->GetBrickSize();
    m_uiBrickExpand = BrickUtils::instance()->get_brick_expand();
    m_uiBrickDim[0] = 1;
    m_uiBrickDim[1] = 1;
    m_uiBrickDim[2] = 1;
}

BrickPool::~BrickPool()
{

}

void BrickPool::get_brick_dim(unsigned int (&uiBrickDim)[3])
{
    memcpy(uiBrickDim , m_uiBrickDim , sizeof(unsigned int)*3);
}

void BrickPool::set_brick_size( unsigned int uiBrickSize )
{
    RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolume);
    m_uiBrickSize= uiBrickSize;
    BrickUtils::instance()->get_brick_dim(m_pVolume->m_uiDim , m_uiBrickDim , m_uiBrickSize);
}

void BrickPool::set_brick_expand( unsigned int uiBrickExpand )
{
    m_uiBrickExpand = uiBrickExpand;
}

void BrickPool::set_volume( std::shared_ptr<ImageData> pImgData )
{
    m_pVolume = pImgData;
}

void BrickPool::set_mask( std::shared_ptr<ImageData> pImgData )
{
    m_pMask = pImgData;
}


BrickCorner* BrickPool::get_brick_corner()
{
    return m_pBrickCorner.get();
}

BrickUnit* BrickPool::get_volume_brick_unit()
{
    return m_pVolumeBrickUnit.get();
}

BrickUnit* BrickPool::get_mask_brick_unit()
{
    return m_pMaskBrickUnit.get();
}

VolumeBrickInfo* BrickPool::get_volume_brick_info()
{
    return m_pVolumeBrickInfo.get();
}

MaskBrickInfo* BrickPool::get_mask_brick_info( const std::vector<unsigned char>& vecVisLabels )
{
    LabelKey key(vecVisLabels);
    auto it = m_mapMaskBrickInfos.find(key);
    if (it == m_mapMaskBrickInfos.end())
    {
        return nullptr;
    }
    else
    {
        return it->second.get();
    }
}

void BrickPool::calculate_volume_brick()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolume);
        BrickUtils::instance()->get_brick_dim(m_pVolume->m_uiDim , m_uiBrickDim , m_uiBrickSize);
        const unsigned int uiBrickCount = m_uiBrickDim[0]*m_uiBrickDim[1]*m_uiBrickDim[2];
        m_pBrickCorner.reset(new BrickCorner[uiBrickCount]);
        m_pVolumeBrickUnit.reset(new BrickUnit[uiBrickCount]);
        m_pVolumeBrickInfo.reset(new VolumeBrickInfo[uiBrickCount]);

        std::cout << "\n<><><><><><><><><><><><><>\n";
        std::cout << "Brick pool info : \n";
        std::cout << "Volume dimension : " << m_pVolume->m_uiDim[0] << " " << m_pVolume->m_uiDim[1] << " "<<m_pVolume->m_uiDim[2] << std::endl;
        std::cout << "Brick size : " << m_uiBrickSize << std::endl;
        std::cout << "Brick expand : " << m_uiBrickExpand << std::endl;
        std::cout << "Brick dimension : " << m_uiBrickDim[0] << " " << m_uiBrickDim[1] << " "<<m_uiBrickDim[2] << std::endl;
        std::cout << "Brick count : " << uiBrickCount << std::endl; 
        std::cout << "Calculate concurrency : " << Concurrency::instance()->get_app_concurrency() << std::endl;

        BrickGenerator brickGen;
        clock_t  t0 = clock();
        brickGen.calculate_brick_corner(m_pVolume , m_uiBrickSize , m_uiBrickExpand , m_pBrickCorner.get());
        clock_t  t1 = clock();
        std::cout << "Calculate brick corner cost : " << double(t1 - t0) << "ms.\n";

        brickGen.calculate_brick_unit(m_pVolume, m_pBrickCorner.get()  , m_uiBrickSize, m_uiBrickExpand ,  m_pVolumeBrickUnit.get());
        clock_t  t2 = clock();
        std::cout << "Calculate volume brick unit cost : " << double(t2 - t1) << "ms.\n";

        if (CPU == Configuration::instance()->get_processing_unit_type())
        {
            CPUVolumeBrickInfoGenerator brickInfoGen;
            brickInfoGen.calculate_brick_info(m_pVolume , m_uiBrickSize , m_uiBrickExpand , 
                m_pBrickCorner.get() , m_pVolumeBrickUnit.get() , m_pVolumeBrickInfo.get());
            clock_t  t3 = clock();
            std::cout << "Calculate volume brick info cost : " << double(t3 - t2) << "ms.\n";
        }
        else
        {

        }

        std::cout << "<><><><><><><><><><><><><>\n";
        

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

void BrickPool::calculate_mask_brick()
{

}

void BrickPool::update_mask_brick(unsigned int (&uiBegin)[3] , unsigned int (&uiEnd)[3])
{

}

MED_IMAGING_END_NAMESPACE