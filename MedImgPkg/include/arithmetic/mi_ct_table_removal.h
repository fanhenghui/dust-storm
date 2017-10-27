#ifndef MEDIMGARITHMETIC_MI_CT_TABLE_REMOVAL_H
#define MEDIMGARITHMETIC_MI_CT_TABLE_REMOVAL_H

#include <memory>
#include <time.h>
#include "util/mi_file_util.h"
#include "arithmetic/mi_arithmetic_export.h"
#include "arithmetic/mi_vector3.h"
#include "arithmetic/mi_arithmetic_logger.h"
#include "arithmetic/mi_morphology.h"

MED_IMG_BEGIN_NAMESPACE

template<class T>
class CTTableRemoval
{
public:
    CTTableRemoval(): _data_ref(nullptr), _mask_ref(nullptr), _target_label(1) {
        _dim[0] = 0;
        _dim[1] = 0;
        _dim[2] = 0;
        _image_orientation[0] = Vector3(1,0,0);
        _image_orientation[1] = Vector3(0,1,0);
        _image_orientation[2] = Vector3(0,0,1);
        _body_threshold = -200.0f;
        _intercept = 0;
        _slope = 1;
    };

    ~CTTableRemoval() {}

    void set_dim(const unsigned int (&dim)[3]) {
        memcpy(_dim, dim, 3 * sizeof(unsigned int));
    }

    void set_data_ref(T* data_array) {
        _data_ref = data_array;
    }

    void set_mask_ref(unsigned char* mask_array) {
        _mask_ref = mask_array;
    }

    void set_target_label(unsigned char label) {
        _target_label = label;
    }

    void set_min_scalar(float min_scalar) {
        _min_scalar = min_scalar;
    }

    void set_max_scalar(float max_scalar) {
        _max_scalar = max_scalar;
    }

    void set_image_orientation(const Vector3 (&ori)[3]) {
        _image_orientation[0] = ori[0];
        _image_orientation[1] = ori[1];
        _image_orientation[2] = ori[2];
    }

    void set_intercept(float intercept) {
        _intercept = intercept;
    }

    void set_slope(float slope) {
        _slope = slope;
    }

    void set_body_threshold(float th) {
        _body_threshold = th;
    }

    void remove();

private:
    int check_axial();
    void to_sagittal(unsigned char* axial_mask, unsigned char* sagittal_mask, int ori_axial_type);
    void back_t0_original(unsigned char* axial_mask, unsigned char* sagittal_mask, int ori_axial_type);

private:
    unsigned int _dim[3];
    T* _data_ref;
    unsigned char* _mask_ref;
    unsigned char _target_label;
    float _min_scalar;
    float _max_scalar;
    float _intercept;
    float _slope;
    float _body_threshold;
    Vector3 _image_orientation[3];
};

#include "arithmetic/mi_ct_table_removal.inl"

MED_IMG_END_NAMESPACE

#endif