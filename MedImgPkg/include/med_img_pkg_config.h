#ifndef MEDIMGPKG_MED_IMG_PKG_CONFIG_H
#define MEDIMGPKG_MED_IMG_PKG_CONFIG_H

#ifndef MED_IMG_NAMESPACE
#define MED_IMG_NAMESPACE                 medical_imaging
#endif

#ifndef MED_IMG_BEGIN_NAMESPACE
#define MED_IMG_BEGIN_NAMESPACE            \
namespace MED_IMG_NAMESPACE           {    /* begin namespace medical_imaging */
#endif
#ifndef MED_IMG_END_NAMESPACE
#define MED_IMG_END_NAMESPACE             }    /* end namespace medical_imaging   */
#endif


typedef double Real;

const double DOUBLE_EPSILON = 1e-15;
const float FLOAT_EPSILON = 1e-6f;

#ifndef DISALLOW_COPY
#define DISALLOW_COPY(class_name) class_name(const class_name & );
#endif

#ifndef DISALLOW_ASSIGN
#define DISALLOW_ASSIGN(class_name) void operator = (const class_name & );
#endif

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(class_name)\
    DISALLOW_COPY  (class_name)\
    DISALLOW_ASSIGN(class_name)
#endif


#endif