#include "mi_brick_define.h"

MED_IMAGING_BEGIN_NAMESPACE

bool operator <(const LabelKey& left, const LabelKey& right)
{
    return left.key < right.key;
}

MED_IMAGING_END_NAMESPACE