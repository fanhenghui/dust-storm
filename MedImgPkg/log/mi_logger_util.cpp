#include "log/mi_logger_util.h"
#include "log/mi_logger.h"

MED_IMG_BEGIN_NAMESPACE

void LoggerUtil::log(SeverityLevel lvl, const std::string& module, const std::string& msg) {
    MI_LOG(lvl) << "[" << module <<  "] " << msg;
}

MED_IMG_END_NAMESPACE