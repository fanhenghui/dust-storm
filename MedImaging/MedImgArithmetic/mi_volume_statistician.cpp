#include "mi_volume_statistician.h"

MED_IMAGING_BEGIN_NAMESPACE


bool Arithmetic_Export operator==(const IntensityInfo& l , const IntensityInfo& r)
{
    if (l._num == r._num && 
        abs(l._min - r._min) < DOUBLE_EPSILON &&
        abs(l._max - r._max) < DOUBLE_EPSILON &&
        abs(l._mean - r._mean) < DOUBLE_EPSILON &&
        abs(l._var - r._var) < DOUBLE_EPSILON &&
        abs(l._std - r._std) < DOUBLE_EPSILON)
    {
        return true;
    }
    else
    {
        return false;
    }
}

MED_IMAGING_END_NAMESPACE