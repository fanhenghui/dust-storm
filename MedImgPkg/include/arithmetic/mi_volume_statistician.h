#ifndef MEDIMGARITHMETIC_MI_VOLUME_STATISTICIAN_H
#define MEDIMGARITHMETIC_MI_VOLUME_STATISTICIAN_H

#include "arithmetic/mi_arithmetic_export.h"
#include "arithmetic/mi_arithmetic_utils.h"
#include "arithmetic/mi_ellipsoid.h"

MED_IMG_BEGIN_NAMESPACE

struct IntensityInfo {
    unsigned int num;
    double min;
    double max;
    double mean;
    double var;
    double std;

    IntensityInfo() : num(0), min(0), max(0), mean(0), var(0), std(0) {};

    IntensityInfo(unsigned int num0, double min0, double max0, double mean0,
                  double var0, double std0)
        : num(num0), min(min0), max(max0), mean(mean0), var(var0),
          std(std0) {};
};

bool Arithmetic_Export operator==(const IntensityInfo& l,
                                  const IntensityInfo& r);

template <class T> class VolumeStatistician {
public:
    VolumeStatistician() : _data_ref(nullptr), _mask_ref(nullptr) {
        _dim[0] = 0;
        _dim[1] = 0;
        _dim[2] = 0;

        memset(_target_labels, 0, sizeof(_target_labels));
    };

    ~VolumeStatistician() {};

    void set_dim(const unsigned int (&dim)[3]) {
        memcpy(_dim, dim, 3 * sizeof(unsigned int));
    }

    void set_data_ref(T* data_array) {
        _data_ref = data_array;
    }

    void set_mask_ref(unsigned char* mask_array) {
        _mask_ref = mask_array;
    }

    void set_target_labels(const std::vector<unsigned char>& labels) {
        memset(_target_labels, 0, sizeof(_target_labels));

        for (auto it = labels.begin(); it != labels.end(); ++it) {
            _target_labels[*it] = 1;
        }
    }

    void get_intensity_analysis(const Ellipsoid& ellipsiod, unsigned int& num,
                                double& min, double& max, double& mean,
                                double& var, double& std);

private:
    unsigned int _dim[3]; // dim[2] could be 0
    T* _data_ref;
    unsigned char* _mask_ref;
    unsigned char _target_labels[256];
};

#include "arithmetic/mi_volume_statistician.inl"

MED_IMG_END_NAMESPACE

#endif