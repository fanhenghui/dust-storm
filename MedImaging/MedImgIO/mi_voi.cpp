#include "mi_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

bool operator==(const VOISphere& l , const VOISphere& r)
{
    return l.center == r.center && abs(l.diameter - r.diameter) < DOUBLE_EPSILON;
}

MED_IMAGING_END_NAMESPACE