#include "mi_brick_define.h"

MED_IMG_BEGIN_NAMESPACE

bool operator <(const LabelKey& left, const LabelKey& right)
{
    return left.key < right.key;
}

MED_IMG_END_NAMESPACE