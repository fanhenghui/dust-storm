#ifndef MED_IMAGING_ARITHMETIC_SAMPLER_H_
#define MED_IMAGING_ARITHMETIC_SAMPLER_H_

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

template<class T>
class Sampler
{
public:
    Sampler() {};

    ~Sampler() {};

    inline float sample_1d_nearst(float idx , unsigned int len , T* data) const;

    inline float sample_1d_linear(float idx , unsigned int len , T* data)const;

    inline float sample_2d_nearst(float x , float y , unsigned int uiWidth , unsigned int uiHeight , T* data)const;

    inline float sample_2d_linear(float x , float y , unsigned int uiWidth , unsigned int uiHeight , T* data)const;

    inline float sample_2d_cubic(float x , float y , unsigned int uiWidth , unsigned int uiHeight , T* data)const;

    inline float sample_3d_nearst(float x , float y , float z , unsigned int uiWidth , unsigned int uiHeight , unsigned int uiDepth , T* data)const;

    inline float sample_3d_linear(float x , float y , float z , unsigned int uiWidth , unsigned int uiHeight , unsigned int uiDepth , T* data)const;

    inline float sample_3d_cubic(float x , float y , float z , unsigned int uiWidth , unsigned int uiHeight , unsigned int uiDepth , T* data)const;

};

#include "mi_sampler.inl"

MED_IMAGING_END_NAMESPACE



#endif