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

    inline float Sample1DNearst(float idx , unsigned int len , T* data) const;

    inline float Sample1DLinear(float idx , unsigned int len , T* data)const;

    inline float Sample2DNearst(float x , float y , unsigned int uiWidth , unsigned int uiHeight , T* data)const;

    inline float Sample2DLinear(float x , float y , unsigned int uiWidth , unsigned int uiHeight , T* data)const;

    inline float Sample2DCubic(float x , float y , unsigned int uiWidth , unsigned int uiHeight , T* data)const;

    inline float Sample3DNearst(float x , float y , float z , unsigned int uiWidth , unsigned int uiHeight , unsigned int uiDepth , T* data)const;

    inline float Sample3DLinear(float x , float y , float z , unsigned int uiWidth , unsigned int uiHeight , unsigned int uiDepth , T* data)const;

    inline float Sample3DCubic(float x , float y , float z , unsigned int uiWidth , unsigned int uiHeight , unsigned int uiDepth , T* data)const;

};

#include "mi_sampler.inl"

MED_IMAGING_END_NAMESPACE



#endif