#include "util/mi_logger.h"

using namespace medical_imaging;

int TestLogger(int argc, char* argv[]) {
    Logger::instance()->init();
    //T t;
    //t.init2();
    MI_LOG(TRACE) << "hello world TRACE";
    MI_LOG(DEBUG) << "hello world DEBUG";
    MI_LOG(INFO) << "hello world INFO";
    MI_LOG(WARNING) << "hello world WARNING";
    MI_LOG(FATAL) << "hello world FATAL";

    for (int i= 0 ; i< 2000 ; ++i)
    {
        MI_LOG(FATAL) << "test rotation.";
    }
    MI_LOG(WARNING) << "ENDENDNED";
    MI_UTIL_LOG(FATAL) <<"util module testing";

    return 1;
}