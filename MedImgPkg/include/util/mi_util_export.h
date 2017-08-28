#ifndef MEDIMGUTIL_MI_UTIL_EXPORT_H
#define MEDIMGUTIL_MI_UTIL_EXPORT_H

#include "med_img_pkg_config.h"

MED_IMG_BEGIN_NAMESPACE 

#ifdef WIN32
#ifdef MEDIMGUTIL_EXPORTS
#define Util_Export __declspec(dllexport)
#else
#define Util_Export __declspec(dllimport)
#endif
#else
#define Util_Export
#endif

MED_IMG_END_NAMESPACE

#endif