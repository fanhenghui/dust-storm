#include "mi_brick_pool.h"

#include "MedImgUtil/mi_configuration.h"

#include "MedImgIO/mi_image_data.h"

#include "mi_brick_utils.h"
#include "mi_brick_generator.h"
#include "mi_brick_info_generator.h"

MED_IMG_BEGIN_NAMESPACE

BrickPool::BrickPool()
{
    _brick_size = BrickUtils::instance()->GetBrickSize();
    _brick_expand = BrickUtils::instance()->get_brick_expand();
    _brick_dim[0] = 1;
    _brick_dim[1] = 1;
    _brick_dim[2] = 1;
}

BrickPool::~BrickPool()
{

}

void BrickPool::get_brick_dim(unsigned int (&brick_dim)[3])
{
    memcpy(brick_dim , _brick_dim , sizeof(unsigned int)*3);
}

void BrickPool::set_brick_size( unsigned int brick_size )
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_data);
    _brick_size= brick_size;
    BrickUtils::instance()->get_brick_dim(_volume_data->_dim , _brick_dim , _brick_size);
}

void BrickPool::set_brick_expand( unsigned int brick_expand )
{
    _brick_expand = brick_expand;
}

void BrickPool::set_volume( std::shared_ptr<ImageData> image_data )
{
    _volume_data = image_data;
}

void BrickPool::set_mask( std::shared_ptr<ImageData> image_data )
{
    _mask_data = image_data;
}


BrickCorner* BrickPool::get_brick_corner()
{
    return _brick_corner_array.get();
}

BrickUnit* BrickPool::get_volume_brick_unit()
{
    return _volume_brick_unit_array.get();
}

BrickUnit* BrickPool::get_mask_brick_unit()
{
    return _mask_brick_unit_array.get();
}

VolumeBrickInfo* BrickPool::get_volume_brick_info()
{
    return _volume_brick_info_array.get();
}

MaskBrickInfo* BrickPool::get_mask_brick_info( const std::vector<unsigned char>& vis_labels )
{
    LabelKey key(vis_labels);
    auto it = _mask_brick_info_array_set.find(key);
    if (it == _mask_brick_info_array_set.end())
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
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume_data);
        BrickUtils::instance()->get_brick_dim(_volume_data->_dim , _brick_dim , _brick_size);
        const unsigned int brick_count = _brick_dim[0]*_brick_dim[1]*_brick_dim[2];
        _brick_corner_array.reset(new BrickCorner[brick_count]);
        _volume_brick_unit_array.reset(new BrickUnit[brick_count]);
        _volume_brick_info_array.reset(new VolumeBrickInfo[brick_count]);

        std::cout << "\n<><><><><><><><><><><><><>\n";
        std::cout << "Brick pool info : \n";
        std::cout << "Volume dimension : " << _volume_data->_dim[0] << " " << _volume_data->_dim[1] << " "<<_volume_data->_dim[2] << std::endl;
        std::cout << "Brick size : " << _brick_size << std::endl;
        std::cout << "Brick expand : " << _brick_expand << std::endl;
        std::cout << "Brick dimension : " << _brick_dim[0] << " " << _brick_dim[1] << " "<<_brick_dim[2] << std::endl;
        std::cout << "Brick count : " << brick_count << std::endl; 

        BrickGenerator brickGen;
        clock_t  t0 = clock();
        brickGen.calculate_brick_corner(_volume_data , _brick_size , _brick_expand , _brick_corner_array.get());
        clock_t  t1 = clock();
        std::cout << "Calculate brick corner cost : " << double(t1 - t0) << "ms.\n";

        brickGen.calculate_brick_unit(_volume_data, _brick_corner_array.get()  , _brick_size, _brick_expand ,  _volume_brick_unit_array.get());
        clock_t  t2 = clock();
        std::cout << "Calculate volume brick unit cost : " << double(t2 - t1) << "ms.\n";

        if (CPU == Configuration::instance()->get_processing_unit_type())
        {
            CPUVolumeBrickInfoGenerator brick_info_generator;
            brick_info_generator.calculate_brick_info(_volume_data , _brick_size , _brick_expand , 
                _brick_corner_array.get() , _volume_brick_unit_array.get() , _volume_brick_info_array.get());
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

void BrickPool::update_mask_brick(unsigned int (&begin)[3] , unsigned int (&end)[3])
{

}

MED_IMG_END_NAMESPACE