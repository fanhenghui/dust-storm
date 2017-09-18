#ifndef MEDIMGLOG_MI_LOGGER_UTIL_H
#define MEDIMGLOG_MI_LOGGER_UTIL_H

#include <string>
#include "log/mi_logger_export.h"
#include "log/mi_logger_define.h"

MED_IMG_BEGIN_NAMESPACE

class LoggerUtil {
public:
    Log_Export static void log(SeverityLevel lvl, const std::string& module, const std::string& msg);
protected:
private:
};
 
MED_IMG_END_NAMESPACE

#endif

