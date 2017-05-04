#include <iostream>
#include <cassert>
#include "mi_main_window.h"
#include <QtGui/QApplication>
#include "MedImgCommon/mi_common_exception.h"

//TODO this will be configured by config file
//#ifdef _DEBUG
#pragma comment( linker, "/subsystem:\"console\" /entry:\"mainCRTStartup\"" )
//#endif


int main(int argc, char *argv[])
{
    try
    {
        QApplication a(argc, argv);
        //a.doubleClickInterval(150);

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
