#ifndef MED_IMG_BRICK_UTILS_H
#define MED_IMG_BRICK_UTILS_H

#include "mi_brick_utils.h"

MED_IMG_BEGIN_NAMESPACE

boost::mutex BrickUtils::_s_mutex;

BrickUtils* BrickUtils::_s_instance = nullptr;

BrickUtils::~BrickUtils()
{

}

BrickUtils::BrickUtils():_brick_size(32),_brick_expand(2)
{

}

BrickUtils* BrickUtils::instance()
{
    if (nullptr == _s_instance)
    {
        boost::unique_lock<boost::mutex> locker(_s_mutex);
        if (nullptr == _s_instance)
        {
            _s_instance = new BrickUtils();
        }
    }
    return _s_instance;
}

void BrickUtils::set_brick_size( unsigned int size )
{
    _brick_size = size;
}

void BrickUtils::set_brick_expand( unsigned int size )
{
    _brick_expand = size;
}

unsigned int BrickUtils::GetBrickSize()
{
    return _brick_size;
}

unsigned int BrickUtils::get_brick_expand()
{
    return _brick_expand;
}

void BrickUtils::get_brick_dim( const unsigned int(&volume_dim)[3] , unsigned int(&brick_dim)[3] , unsigned int brick_size)
{
    for (int i = 0 ; i< 3 ; ++i)
    {
        brick_dim[i] = (unsigned int)floor((float)volume_dim[i]/(float)brick_size);
    }
}






MED_IMG_END_NAMESPACE
#endif