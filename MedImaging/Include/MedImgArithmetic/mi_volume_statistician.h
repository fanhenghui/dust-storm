#ifndef MED_IMAGING_ARRAY_STATISTICIAN_H_
#define MED_IMAGING_ARRAY_STATISTICIAN_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"
#include "MedImgArithmetic/mi_sphere.h"
#include "MedImgArithmetic/mi_ellipsoid.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

struct IntensityInfo
{
    unsigned int _num;
    double _min;
    double _max;
    double _mean;
    double _var;
    double _std;

    IntensityInfo():_num(0),_min(0),_max(0),_mean(0),_var(0),_std(0)
    {};

    IntensityInfo(unsigned int num0 , double min0 , double max0 , double mean0 , double var0 , double std0):
    _num(num0),_min(min0),_max(max0),_mean(mean0),_var(var0),_std(std0)
    {};
};

bool Arithmetic_Export operator == (const IntensityInfo& l  , const IntensityInfo& r);

template<class T>
class VolumeStatistician
{
public:
    VolumeStatistician() {};
    ~VolumeStatistician() {};
 
    void get_intensity_analysis(const unsigned int (&dim)[3] , T* data_array , const Sphere& sphere, 
        unsigned int& num , double& min , double& max , double& mean , double& var, double& std);

    void get_intensity_analysis(const unsigned int (&dim)[3] , T* data_array , const Ellipsoid& ellipsiod, 
        unsigned int& num , double& min , double& max , double& mean , double& var, double& std);
};

#include "MedImgArithmetic/mi_volume_statistician.inl"

MED_IMAGING_END_NAMESPACE

#endif