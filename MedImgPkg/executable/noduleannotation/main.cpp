#include <iostream>
#include <cassert>
#include "mi_main_window.h"
#include <QtGui/QApplication>
#include "util/mi_exception.h"
#include "log/mi_logger.h"

//TODO this will be configured by config file
#ifdef _DEBUG
#pragma comment( linker, "/subsystem:\"console\" /entry:\"mainCRTStartup\"" )
#endif

using namespace medical_imaging;

int main(int argc, char *argv[])
{
    try
    {
        medical_imaging::Logger::instance()->initialize();

        MI_LOG(MI_TRACE) << "QT hello world TRACE";
        MI_LOG(MI_DEBUG) << "QT hello world DEBUG";
        MI_LOG(MI_INFO) << "QT hello world INFO";
        MI_LOG(MI_WARNING) << "QT hello world WARNING";
        MI_LOG(MI_FATAL) << "QT hello world FATAL";

        QApplication a(argc, argv);
        //std::cout << a.doubleClickInterval() << std::endl;

        NoduleAnnotation w;
        w.show();
        return a.exec();
    }
    catch (const medical_imaging::Exception& e)
    {
        std::cout << e.what();
        assert(false);
        return 0;
    }
    
}
