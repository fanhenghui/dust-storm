#ifndef MEDIMGDBS_MI_DB_SERVER_LOGGER_H
#define MEDIMGDBS_MI_DB_SERVER_LOGGER_H

#include "log/mi_logger.h"

#define MI_DBSERVER_LOG(sev) MI_LOG(sev) << "[DBSERVER] "
#define MI_DBSERVER_LOG_DETAIL(sev) MI_UTIL_LOG(sev) << MI_DETAIL_FORMAT  

#endif