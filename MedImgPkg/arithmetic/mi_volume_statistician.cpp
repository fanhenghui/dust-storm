#include "mi_volume_statistician.h"

MED_IMG_BEGIN_NAMESPACE

bool Arithmetic_Export operator==(const IntensityInfo &l,
                                  const IntensityInfo &r) {
    if (l._num == r._num && fabs(l._min - r._min) < DOUBLE_EPSILON &&
            fabs(l._max - r._max) < DOUBLE_EPSILON &&
            fabs(l._mean - r._mean) < DOUBLE_EPSILON &&
            fabs(l._var - r._var) < DOUBLE_EPSILON &&
            fabs(l._std - r._std) < DOUBLE_EPSILON) {
        return true;
    } else {
        return false;
    }
}

MED_IMG_END_NAMESPACE