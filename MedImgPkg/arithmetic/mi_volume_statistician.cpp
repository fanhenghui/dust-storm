#include "mi_volume_statistician.h"

MED_IMG_BEGIN_NAMESPACE

bool Arithmetic_Export operator==(const IntensityInfo& l,
                                  const IntensityInfo& r) {
    if (l.num == r.num && fabs(l.min - r.min) < DOUBLE_EPSILON &&
            fabs(l.max -  r.max) < DOUBLE_EPSILON &&
            fabs(l.mean - r.mean) < DOUBLE_EPSILON &&
            fabs(l.var -  r.var) < DOUBLE_EPSILON &&
            fabs(l.std -  r.std) < DOUBLE_EPSILON) {
        return true;
    } else {
        return false;
    }
}

MED_IMG_END_NAMESPACE