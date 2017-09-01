#ifndef MEDIMGARITHMETIC_MI_SEGMENT_THRESHOLD_H
#define MEDIMGARITHMETIC_MI_SEGMENT_THRESHOLD_H

#include "arithmetic/mi_sampler.h"
#include "arithmetic/mi_segment_interface.h"

MED_IMG_BEGIN_NAMESPACE

template <class T> class SegmentThreshold : public ISegment<T> {
public:
    enum ThresholdType {
        Center,
        Otsu,
    };

public:
    SegmentThreshold() {};
    virtual ~SegmentThreshold() {};

    void segment(const Ellipsoid& region_ellipsoid, T threshold);

    void segment(T threshold);

    void segment_auto_threshold(const Ellipsoid& region_ellipsoid,
                                ThresholdType type = Otsu);

public:
    T get_threshold_otsu_i(const Ellipsoid& region_ellipsoid);

    T get_threshold_center_i(const Ellipsoid& region_ellipsoid);

    T get_threshold_otsu_i();
};

#include "arithmetic/mi_segment_threshold.inl"

MED_IMG_END_NAMESPACE

#endif