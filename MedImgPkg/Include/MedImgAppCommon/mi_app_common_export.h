#ifndef MED_IMG_APP_COMMON_EXPORT_H
#define MED_IMG_APP_COMMON_EXPORT_H

#include "med_img_pkg_config.h"
#include "MedImgUtil/mi_exception.h"

MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32
    #ifdef MEDIMGAPPCOMMON_EXPORTS
    #define AppCommon_Export __declspec(dllexport)
    #else
    #define AppCommon_Export __declspec(dllimport)
    #endif
#else
    #define AppCommon_Export
#endif

#ifndef APPCOMMON_THROW_EXCEPTION
#define APPCOMMON_THROW_EXCEPTION(desc) THROW_EXCEPTION("AppCommon" , desc);
#endif

MED_IMG_END_NAMESPACE
#endif