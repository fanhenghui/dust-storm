#ifndef MI_RAW_DATA_IMPORT_H_
#define MI_RAW_DATA_IMPORT_H_

#include <string>
#include <QDialog>
#include "ui_mi_raw_data_dialog.h"

namespace medical_imaging
{
    class ImageData;
    class ImageDataHeader;
}

class RawDataImportDlg : public QDialog
{
    Q_OBJECT
public:
    RawDataImportDlg(QWidget *parent = 0, Qt::WindowFlags f = 0);
    virtual ~RawDataImportDlg();

Q_SIGNALS:
    void raw_data_imported(
        std::shared_ptr<medical_imaging::ImageData> img_data ,
        std::shared_ptr<medical_imaging::ImageDataHeader> data_header);

protected Q_SLOTS:
    void slot_press_btn_browse_path_i();
    void slot_press_btn_import_i();
    void slot_press_btn_cancel_i();

private:
    Ui::RawDataImportDlgUI _ui;
};

#endif