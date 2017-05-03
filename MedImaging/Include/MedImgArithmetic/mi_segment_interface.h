#ifndef MED_IMAGING_SEGMENT_INTERFACE_H_
#define MED_IMAGING_SEGMENT_INTERFACE_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

template<class T>
class ISegment
{
public:
    ISegment():_data_ref(nullptr),_mask_ref(nullptr),_target_label(1)
    {
        _dim[0] = 0;
        _dim[1] = 0;
        _dim[2] = 0;
    };

    virtual ISegment() {};

    void set_dim(const unsigned int (&dim)[3] )
    {
        memcpy(_dim , dim ,3*sizeof*(unsigned int));
    }

    void set_data_ref(T* data_array)
    {
        _data_ref = data_array;
    }

    void set_mask_ref(unsigned char* mask_array)
    {
        _mask_ref = mask_array;
    }

    void set_target_label(unsigned char label)
    {
        _target_label = label;
    }

protected:
    unsigned int _dim[3];//dim[2] could be 0
    T* _data_ref;
    unsigned char* _mask_ref;
    unsigned char _target_label;
};

MED_IMAGING_END_NAMESPACE

#endif