#ifndef MEDIMGUTIL_MI_UTIL_LOGGER_H
#define MEDIMGUTIL_MI_UTIL_LOGGER_H

#include "log/mi_logger.h"

#define MI_UTIL_LOG(sev) MI_LOG(sev) << "[UTIL] "
#define MI_UTIL_LOG_DETAIL(sev) MI_UTIL_LOG(sev) << MI_DETAIL_FORMAT  
#endif