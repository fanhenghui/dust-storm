#ifndef MEDIMGREVIEW_MI_REVIEW_LOGGER_H
#define MEDIMGREVIEW_MI_REVIEW_LOGGER_H

#include "log/mi_logger.h"

#define MI_REVIEW_LOG(sev) MI_LOG(sev) << "[REVIEW] "
#define MI_REVIEW_LOG_DETAIL(sev) MI_UTIL_LOG(sev) << MI_DETAIL_FORMAT  

#endif