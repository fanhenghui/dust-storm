#ifndef MEDIMGLOG_MI_LOG_EXPORT_H
#define MEDIMGLOG_MI_LOG_EXPORT_H

#include "med_img_pkg_config.h"

MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32
#ifdef MEDIMGLOG_EXPORTS
#define Log_Export __declspec(dllexport)
#else
#define Log_Export __declspec(dllimport)
#endif
#else
#define Log_Export
#endif

MED_IMG_END_NAMESPACE

#endif