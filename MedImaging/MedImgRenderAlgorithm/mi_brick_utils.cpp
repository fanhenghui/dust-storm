#ifndef MED_IMAGING_BRICK_UTILS_H
#define MED_IMAGING_BRICK_UTILS_H

#include "mi_brick_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

boost::mutex BrickUtils::_mutex;

BrickUtils* BrickUtils::_s_instance = nullptr;

BrickUtils::~BrickUtils()
{

}

BrickUtils::BrickUtils():m_uiBrickSize(32),m_uiBrickExpand(2)
{

}

BrickUtils* BrickUtils::instance()
{
    if (nullptr == _s_instance)
    {
        boost::unique_lock<boost::mutex> locker(_mutex);
        if (nullptr == _s_instance)
        {
            _s_instance = new BrickUtils();
        }
    }
    return _s_instance;
}

void BrickUtils::set_brick_size( unsigned int uiSize )
{
    m_uiBrickSize = uiSize;
}

void BrickUtils::set_brick_expand( unsigned int uiSize )
{
    m_uiBrickExpand = uiSize;
}

unsigned int BrickUtils::GetBrickSize()
{
    return m_uiBrickSize;
}

unsigned int BrickUtils::get_brick_expand()
{
    return m_uiBrickExpand;
}

void BrickUtils::get_brick_dim( const unsigned int(&uiVolumeDim)[3] , unsigned int(&uiBrickDim)[3] , unsigned int uiBrickSize)
{
    for (int i = 0 ; i< 3 ; ++i)
    {
        uiBrickDim[i] = (unsigned int)floor((float)uiVolumeDim[i]/(float)uiBrickSize);
    }
}






MED_IMAGING_END_NAMESPACE
#endif