#ifndef MED_IMAGING_COMMON_H
#define MED_IMAGING_COMMON_H

#include "med_imaging_config.h"

MED_IMAGING_BEGIN_NAMESPACE

#ifdef WIN32

#ifdef MEDIMGCOMMON_EXPORTS
#define Common_Export __declspec(dllexport)
#else
#define Common_Export __declspec(dllimport)
#endif

#else
#define Common_Export
#endif

#pragma warning(disable: 4251)

MED_IMAGING_END_NAMESPACE

#endif