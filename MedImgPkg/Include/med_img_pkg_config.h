#pragma once

#ifndef MED_IMG_NAMESPACE
#define MED_IMG_NAMESPACE                 medical_imaging
#endif

#ifndef MED_IMG_BEGIN_NAMESPACE
#define MED_IMG_BEGIN_NAMESPACE           \
    namespace MED_IMG_NAMESPACE           {    /* begin namespace medical_imaging */
#endif
#ifndef MED_IMG_END_NAMESPACE
#define MED_IMG_END_NAMESPACE             }    /* end namespace medical_imaging   */
#endif


//对于精度，根据Peter Shirley的建议，对于精度较高的光线-对象 相交计算，将采用双精度浮点数；而对着色器计算则采用单精度浮点数计算。
//但是对于医学应用来说，还是选择双精度会更加合适，在需要CPU加速的地方先转换为单精度，再进行运算
typedef double Real;

#define DOUBLE_EPSILON 1e-15
#define FLOAT_EPSILON 1e-6f