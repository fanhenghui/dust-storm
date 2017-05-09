#ifndef MED_IMAGING_THRESHOLD_SEGMENATATION_H_
#define MED_IMAGING_THRESHOLD_SEGMENATATION_H_

#include "MedImgArithmetic/mi_segment_interface.h"
#include "MedImgArithmetic/mi_sampler.h"
//#include "ext/boost/atomic/atomic.hpp"

MED_IMAGING_BEGIN_NAMESPACE


template<class T>
class SegmentThreshold : public ISegment<T>
{
public:
    enum ThresholdType
    {
        Center,
        Otsu,
    };

public:
    SegmentThreshold() {};
    virtual ~SegmentThreshold() {};

    void segment(const Ellipsoid& region_ellipsoid , T threshold);

    void segment_auto_threshold(const Ellipsoid& region_ellipsoid , ThresholdType type = Otsu);

private:
    T get_threshold_otsu_i(const Ellipsoid& region_ellipsoid);

    T get_threshold_center_i(const Ellipsoid& region_ellipsoid);

};

#include "MedImgArithmetic/mi_segment_threshold.inl"

MED_IMAGING_END_NAMESPACE

#endif