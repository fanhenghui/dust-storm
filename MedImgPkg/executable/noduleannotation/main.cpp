#include <iostream>
#include <cassert>
#include "mi_main_window.h"
#include <QtGui/QApplication>
#include "util/mi_common_exception.h"

//TODO this will be configured by config file
#ifdef _DEBUG
#pragma comment( linker, "/subsystem:\"console\" /entry:\"mainCRTStartup\"" )
#endif


int main(int argc, char *argv[])
{
    try
    {
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
