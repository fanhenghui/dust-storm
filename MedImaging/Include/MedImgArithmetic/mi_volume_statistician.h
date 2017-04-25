#ifndef MED_IMAGING_ARRAY_STATISTICIAN_H_
#define MED_IMAGING_ARRAY_STATISTICIAN_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"
#include "MedImgArithmetic/mi_sphere.h"
#include "MedImgArithmetic/mi_ellipsoid.h"

MED_IMAGING_BEGIN_NAMESPACE

template<class T>
class VolumeStatistician
{
public:
    VolumeStatistician() {};
    ~VolumeStatistician() {};
 
    void get_intensity_analysis(const unsigned int (&dim)[3] , T* data_array , const Sphere& sphere, int& num , double& min , double& max , double& mean , double& var, double& std);

    void get_intensity_analysis(const unsigned int (&dim)[3] , T* data_array , const Ellipsoid& ellipsiod, int& num , double& min , double& max , double& mean , double& var, double& std);

protected:
private:
};

#include "MedImgArithmetic/mi_volume_statistician.inl"

MED_IMAGING_END_NAMESPACE

#endif