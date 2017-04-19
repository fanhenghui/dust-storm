#include "login_in_dialog.h"
#include <QTextDocument>
#include <QMessageBox>


LoginInDialog::LoginInDialog(QDialog* parent) : QDialog(parent)
{
    ui.setupUi(this);
    connect(ui.okButton , SIGNAL(clicked()) , this , SLOT(SlotsPushButtonOKClicked()));

    ui.nameEdit->document()->setPlainText(tr("wangrui"));
    ui.passwordEdit->document()->setPlainText(tr("12345"));
}

LoginInDialog::~LoginInDialog()
{

}

void LoginInDialog::SlotsPushButtonOKClicked()
{
    if (ui.nameEdit->document()->toPlainText() == tr("wangrui") &&
        ui.passwordEdit->document()->toPlainText() == tr("12345"))
    {
        accept();
    }
    else
    {
        QMessageBox::warning(this , tr("Waring") , tr("user name or password error!") , QMessageBox::Yes);

        ui.nameEdit->clear();
        ui.passwordEdit->clear();
        ui.nameEdit->setFocus();
    }
}
