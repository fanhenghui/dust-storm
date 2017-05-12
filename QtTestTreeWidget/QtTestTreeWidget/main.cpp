#include "QtTestTreeWidget.h"
#include <QtWidgets/QApplication>

#ifdef _DEBUG
#pragma comment( linker, "/subsystem:\"console\" /entry:\"mainCRTStartup\"" )
#endif

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QtTestTreeWidget w;
    w.show();
    return a.exec();
}
