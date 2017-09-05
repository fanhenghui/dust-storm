#include "mi_setting_dialog.h"

SettingDlg::SettingDlg(QWidget *parent /*= 0*/, Qt::WindowFlags f /*= 0*/)
{
    _ui.setupUi(this);
    this->setWindowFlags(Qt::Dialog | Qt::WindowCloseButtonHint);
    connect(this->_ui.save_btn, SIGNAL(clicked()), this, SLOT(close_window()));
    connect(this->_ui.close_btn, SIGNAL(clicked()), this, SLOT(close_window()));
}

SettingDlg::~SettingDlg()
{

}

void SettingDlg::set_preset_wl(PreSetWLType type, float ww, float wl)
{
    QLineEdit* w_entry = nullptr;
    QLineEdit* l_entry = nullptr;
    switch(type)
    {
    case CT_ABDOMEN:
        w_entry = this->_ui.abdomen_w;
        l_entry = this->_ui.abdomen_l;
        break;
    case CT_LUNGS:
        w_entry = this->_ui.lung_w;
        l_entry = this->_ui.lung_l;
        break;
    case CT_BRAIN:
        w_entry = this->_ui.brain_w;
        l_entry = this->_ui.brain_l;
        break;
    case CT_ANGIO:
        w_entry = this->_ui.angio_w;
        l_entry = this->_ui.angio_l;
        break;
    case CT_BONE:
        w_entry = this->_ui.bone_w;
        l_entry = this->_ui.bone_l;
        break;
    case CT_CHEST:
        w_entry = this->_ui.chest_w;
        l_entry = this->_ui.chest_l;
        break;
    }
    if (w_entry && l_entry)
    {
        w_entry->setText(QString::number(ww));
        l_entry->setText(QString::number(wl));
    }
}

void SettingDlg::close_window()
{
    if (this->sender() == this->_ui.save_btn)
    {
        std::vector<float> all_ww_wl;
        QLineEdit* all_edit[2*CT_Preset_ALL] = {
            this->_ui.abdomen_w,
            this->_ui.abdomen_l,

            this->_ui.lung_w,
            this->_ui.lung_l,

            this->_ui.brain_w,
            this->_ui.brain_l,
            
            this->_ui.angio_w,
            this->_ui.angio_l,

            this->_ui.bone_w,
            this->_ui.bone_l,

            this->_ui.chest_w,
            this->_ui.chest_l,
        };

        for (int i=0; i<CT_Preset_ALL*2; ++i)
        {
            float val = all_edit[i]->text().toFloat();
            all_ww_wl.push_back(val);
        }
        emit save_setting(all_ww_wl);
    }
    this->close();
}
