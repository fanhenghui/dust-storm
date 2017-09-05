#ifndef MI_SETTING_H_
#define MI_SETTING_H_

#include <QDialog>
#include "ui_mi_setting.h"
#include "mi_nodule_anno_config.h"
class SettingDlg : public QDialog
{
    Q_OBJECT
public:
    SettingDlg(QWidget *parent = 0, Qt::WindowFlags f = 0);
    virtual ~SettingDlg();
    void set_preset_wl(PreSetWLType type, float ww, float wl);
    void get_preset_wl(PreSetWLType type, float &ww, float &wl);

Q_SIGNALS:
    void save_setting(std::vector<float>);

private Q_SLOTS:
    void close_window();
private:
    Ui::PreWLSetting _ui;
};

#endif