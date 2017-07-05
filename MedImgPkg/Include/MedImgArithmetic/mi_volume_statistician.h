#ifndef MED_IMG_ARRAY_STATISTICIAN_H_
#define MED_IMG_ARRAY_STATISTICIAN_H_

#include "MedImgArithmetic/mi_arithmetic_export.h"
#include "MedImgArithmetic/mi_ellipsoid.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

#ifdef WIN32
#else
#include <typeinfo> 
#endif

MED_IMG_BEGIN_NAMESPACE

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
    VolumeStatistician():_data_ref(nullptr),_mask_ref(nullptr)
    {
        _dim[0] = 0;
        _dim[1] = 0;
        _dim[2] = 0;

        memset(_target_labels , 0 , sizeof(_target_labels));
    };

    ~VolumeStatistician() {};

    void set_dim(const unsigned int (&dim)[3] )
    {
        memcpy(_dim , dim ,3*sizeof(unsigned int));
    }

    void set_data_ref(T* data_array)
    {
        _data_ref = data_array;
    }

    void set_mask_ref(unsigned char* mask_array)
    {
        _mask_ref = mask_array;
    }

    void set_target_labels(const std::vector<unsigned char>& labels)
    {
        memset(_target_labels , 0 , sizeof(_target_labels));
        for (auto it = labels.begin() ; it != labels.end() ; ++it)
        {
            _target_labels[*it] = 1;
        }
    }

    void get_intensity_analysis(const Ellipsoid& ellipsiod, 
        unsigned int& num , double& min , double& max , double& mean , double& var, double& std);

private:
    unsigned int _dim[3];//dim[2] could be 0
    T* _data_ref;
    unsigned char* _mask_ref;
    unsigned char _target_labels[256];
};

#include "MedImgArithmetic/mi_volume_statistician.inl"

MED_IMG_END_NAMESPACE

#endif