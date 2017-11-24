#ifndef MEDIMGAIS_MI_AI_SERVER_LOGGER_H
#define MEDIMGAIS_MI_AI_SERVER_LOGGER_H

#include "log/mi_logger.h"

#define MI_AISERVER_LOG(sev) MI_LOG(sev) << "[AISERVER] "
#define MI_AISERVER_LOG_DETAIL(sev) MI_UTIL_LOG(sev) << MI_DETAIL_FORMAT  

#endif