#ifndef MEDIMGLOG_MI_LOGGER_DEFINE_H
#define MEDIMGLOG_MI_LOGGER_DEFINE_H

#include "med_img_pkg_config.h"

MED_IMG_BEGIN_NAMESPACE

enum SeverityLevel {
    MI_TRACE,
    MI_DEBUG,
    MI_INFO,
    MI_WARNING,
    MI_ERROR,
    MI_FATAL,
};

MED_IMG_END_NAMESPACE

#endif