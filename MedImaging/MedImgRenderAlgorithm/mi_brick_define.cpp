#include "mi_brick_define.h"

MED_IMAGING_BEGIN_NAMESPACE

bool operator <(const LabelKey& _Left, const LabelKey& _Right)
{
    return _Left.m_sKey < _Right.m_sKey;
}

MED_IMAGING_END_NAMESPACE