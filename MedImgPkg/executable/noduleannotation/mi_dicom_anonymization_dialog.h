#ifndef MI_DICOM_ANONYMIZATION_DIALOG_H_
#define MI_DICOM_ANONYMIZATION_DIALOG_H_

#include <string>
#include <QDialog>
#include "ui_mi_anonymization_dialog.h"

namespace medical_imaging {
class ProgressModel;
}

class DICOMAnonymizationDlg : public QDialog {
    Q_OBJECT
public:
    DICOMAnonymizationDlg(QWidget* parent = 0, Qt::WindowFlags f = 0);
    virtual ~DICOMAnonymizationDlg();

    void set_dicom_series_files(std::vector<std::string>& files);

    void set_progress_model(std::shared_ptr<medical_imaging::ProgressModel> model);

protected:

private slots:
    void slot_press_btn_browse_path_i();
    void slot_press_btn_export_i();
    void slot_press_btn_cancel_i();

private:
    Ui::Form _ui;
    std::vector<std::string> _dicom_series_files;
    std::shared_ptr<medical_imaging::ProgressModel> _model_progress;
};

#endif