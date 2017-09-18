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
        MI_LOG(MI_TRACE) << "nodule annotation start.";
        medical_imaging::Logger::instance()->initialize();

        QApplication a(argc, argv);
        //MI_LOG(MI_DEBUG) << a.doubleClickInterval() << std::endl;

        NoduleAnnotation w;
        w.show();
        return a.exec();
    }
    catch (const medical_imaging::Exception& e)
    {
        MI_LOG(MI_ERROR) << "nodule annotation abort with exception: " << e.what();
        assert(false);
        return 0;
    }
}
