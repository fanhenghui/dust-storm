#include "util/mi_util_logger.h"

using namespace medical_imaging;
//extern std::ostream& operator<<(std::ostream& strm, medical_imaging::SeverityLevel lvl);
int TestLogger(int argc, char* argv[]) {
    Logger::instance()->initialize();
    //T t;
    //t.init2();
    MI_LOG(MI_TRACE) << "hello world TRACE";
    MI_LOG(MI_DEBUG) << "hello world DEBUG";
    MI_LOG(MI_INFO) << "hello world INFO";
    MI_LOG(MI_WARNING) << "hello world WARNING";
    MI_LOG(MI_FATAL) << "hello world FATAL";

    /*for (int i= 0 ; i< 2000 ; ++i)
    {
        MI_LOG(MI_FATAL) << "test rotation.";
    }*/
    MI_LOG(MI_WARNING) << "ENDENDNED";
    MI_UTIL_LOG(MI_FATAL) <<"util module testing";

    return 1;
}