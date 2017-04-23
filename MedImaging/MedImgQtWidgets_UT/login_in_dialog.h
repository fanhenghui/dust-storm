#ifndef LOGIN_IN_DIALOG_H_
#define LOGIN_IN_DIALOG_H_

#include "ui_logini_in_dialog.h"
#include <QDialog>

class LoginInDialog : public QDialog
{
    Q_OBJECT
public:
    LoginInDialog(QDialog* parent = 0);
    virtual ~LoginInDialog();
protected:

private slots:
    void slot_push_button_ok_clicked();

private:
    Ui::Dialog ui;
};

#endif