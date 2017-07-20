#include "mi_aabb.h"

MED_IMAGING_BEGIN_NAMESPACE

AABB::AABB()
{

}

AABB::~AABB()
{

}

bool AABB::operator==(const AABB& aabb) const
{
    return _min == aabb._min && _max == aabb._max;
}

bool AABB::operator!=(const AABB& aabb) const
{
    return _min != aabb._min || _max != aabb._max;
}



AABBUI::AABBUI()
{
    memset(_min , 0 , sizeof(_min));
    memset(_max , 0 , sizeof(_max));
}

AABBUI::AABBUI(const unsigned int (&min0)[3] , const unsigned int (&max0)[3])
{
    memcpy(_min ,min0 , sizeof(_min));
    memcpy(_max ,max0 , sizeof(_max));
}

AABBUI::~AABBUI()
{

}

int AABBUI::Intersect(const AABBUI & aabb)
{
    bool intersects;
    unsigned int pMin[3]={0u, 0u, 0u}, pMax[3] = {1024u, 1024u, 1024u};
    for (unsigned i = 0; i < 3; i++)
    {
        intersects = false;
        if ((aabb._min[i] >= this->_min[i]) && (aabb._min[i] <= this->_max[i]))
        {
            intersects = true;
            pMin[i] = aabb._min[i];
        }
        else if ((this->_min[i] >= aabb._min[i]) && (this->_min[i] <= aabb._max[i]))
        {
            intersects = true;
            pMin[i] = this->_min[i];
        }

        if ((aabb._max[i] >= this->_min[i]) && (aabb._max[i] <= this->_max[i]))
        {
            intersects = true;
            pMax[i] = aabb._max[i];
        }
        else if ((this->_max[i] >= aabb._min[i]) && (this->_max[i] <= aabb._max[i]))
        {
            intersects = true;
            pMax[i] = this->_max[i];
        }
        if (!intersects)
        {
            return 0;
        }
    }

    // OK they did intersect - set the box to be the result
    for (unsigned i = 0; i < 3; i++)
    {
        this->_min[i] = pMin[i];
        this->_max[i] = pMax[i];
    }
    return 1;
}

bool AABBUI::operator==(const AABBUI& aabb) const
{
    return (_min[0] == aabb._min[0] && _min[1] == aabb._min[1] && _min[2] == aabb._min[2] &&
        _max[0] == aabb._max[0] && _max[1] == aabb._max[1] && _max[2] == aabb._max[2]);
}

bool AABBUI::operator!=(const AABBUI& aabb) const
{
    return (_min[0] != aabb._min[0] || _min[1] != aabb._min[1] || _min[2] != aabb._min[2] ||
        _max[0] != aabb._max[0] || _max[1] != aabb._max[1] || _max[2] != aabb._max[2]);
}

void AABBUI::Print()
{
    std::cout << "AABBUI : [ " << _min[0] << " " << _min[1] << " " << _min[2] << " ] , [" << _max[0] << " " << _max[1] << " " << _max[2] <<" ]\n";
}

MED_IMAGING_END_NAMESPACE