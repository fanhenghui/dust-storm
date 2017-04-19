#include "my_main_window.h"
#include <QtGui/QApplication>
#include <Windows.h>
#include "login_in_dialog.h"
#include "paint_dialog.h"

#pragma comment( linker, "/subsystem:\"console\" /entry:\"mainCRTStartup\"" )

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MyMainWindow w;

    LoginInDialog dlg;
    if (dlg.exec() == QDialog::Accepted)
    {
        w.show();
        return a.exec();
    }
    else
    {
        PaintDialog pdlg;
        pdlg.resize(500,500);
        pdlg.exec();
        return 0;
    }
}
