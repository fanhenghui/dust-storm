#include <QtGui/QApplication>

#include "my_window.h"

#pragma comment( linker, "/subsystem:\"console\" /entry:\"mainCRTStartup\"" )

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MyWindow w;
    w.resize(1024,1024);
    w.show();
    return a.exec();
}
