#ifndef MED_IMAGING_THRESHOLD_SEGMENATATION_H_
#define MED_IMAGING_THRESHOLD_SEGMENATATION_H_


#include "MedImgArithmetic/mi_segment_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

template<class T>
class SegmentThreshold : public ISegment<T>
{
public:
    SegmentThreshold() {};
    virtual ~SegmentThreshold() {};

    void segment(const Ellipsoid& region_ellipsoid , T threshold);

protected:
private:
};

#include "MedImgArithmetic/mi_segment_threshold.inl"

MED_IMAGING_END_NAMESPACE

#endif