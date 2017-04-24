#ifndef MED_IMAGING_ARRAY_STATISTICIAN_H_
#define MED_IMAGING_ARRAY_STATISTICIAN_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

template<class T>
class VolumeStatistician
{
public:
    VolumeStatistician();
    ~VolumeStatistician();

    void get_min_max(unsigned int (&dim)[3] ,T* data_array , Sphere& sphere, float& min , float& max);
    void get_mean(unsigned int (&dim)[3] ,T* data_array , Sphere& sphere , float& mean);
    void get_variance(unsigned int (&dim)[3] ,T* data_array , Sphere& sphere , float mean , float variance);
    void get_variance(unsigned int (&dim)[3] ,T* data_array , Sphere& sphere , float& min , float& max);
protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif