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
    return 1;
}