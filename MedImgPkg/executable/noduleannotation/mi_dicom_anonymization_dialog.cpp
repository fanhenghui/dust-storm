#include "mi_dicom_anonymization_dialog.h"

#include <QPushButton>
#include <QFileDialog>
#include <QLineEdit>
#include <QProgressDialog>
#include <QMessageBox>
#include <QApplication>

#include "io/mi_dicom_exporter.h"
#include "io/mi_model_progress.h"

#include "qtpackage/mi_observer_progress.h"

using namespace medical_imaging;

DICOMAnonymizationDlg::DICOMAnonymizationDlg(QWidget *parent /*= 0*/, Qt::WindowFlags f /*= 0*/):
        QDialog(parent , f)
{
    _ui.setupUi(this);

    this->setWindowFlags(Qt::Dialog | Qt::WindowCloseButtonHint);

    connect(_ui.pushButton_browse_path, SIGNAL(pressed()) , this , SLOT(slot_press_btn_browse_path_i()));
    connect(_ui.pushButton_export, SIGNAL(pressed()) , this , SLOT(slot_press_btn_export_i()));
    connect(_ui.pushButton_cancel, SIGNAL(pressed()) , this , SLOT(slot_press_btn_cancel_i()));

    _ui.lineEdit_postfix->setText(tr("_anon.dcm"));
}

DICOMAnonymizationDlg::~DICOMAnonymizationDlg()
{

}

void DICOMAnonymizationDlg::set_dicom_series_files(std::vector<std::string>& files)
{
    _dicom_series_files = files;

    if (!files.empty())
    {
        std::string file0 = files[0];
        file0.substr(0 , 10);
        int sub = file0.size() - 1;
        for (; sub>=0 ; --sub)
        {
            if (file0[sub] == '\\' || file0[sub] == '/')
            {
                break;
            }
        }
        if (0 != sub)
        {
            std::string default_path = file0.substr(0 , sub);
            _ui.lineEdit_path->setText(default_path.c_str());
        }
    }
}

void DICOMAnonymizationDlg::slot_press_btn_browse_path_i()
{
    QString file_folder = QFileDialog::getExistingDirectory(this ,tr("Export path dialog") , _ui.lineEdit_path->text() );
    _ui.lineEdit_path->setText(file_folder);
}

void DICOMAnonymizationDlg::slot_press_btn_export_i()
{
    //1 Select tags
    QCheckBox* checkboxes[11] = 
    {
        _ui.checkBox_name,
        _ui.checkBox_age,
        _ui.checkBox_id,
        _ui.checkBox_weight,
        _ui.checkBox_address,
        _ui.checkBox_sex,
        _ui.checkBox_birth_date,
        _ui.checkBox_birth_time,
        _ui.checkBox_birth_name,
        _ui.checkBox_other_ids,
        _ui.checkBox_other_names
    };

    DcmTagKey tags[11] = 
    {
        DCM_PatientName,
        DCM_PatientAge,
        DCM_PatientID,
        DCM_PatientWeight,
        DCM_PatientAddress,
        DCM_PatientSex,
        DCM_PatientBirthDate,
        DCM_PatientBirthTime,
        DCM_PatientBirthName,
        DCM_OtherPatientIDs,
        DCM_OtherPatientNames
    };

    std::vector<DcmTagKey> anonymous_tags;
    for (int i = 0 ; i < 11; ++i)
    {
        if (checkboxes[i]->isChecked())
        {
            anonymous_tags.push_back(tags[i]);
        }
    }

    //2 Construct export file names
    const std::string custom_postfix( _ui.lineEdit_postfix->text().toLocal8Bit());
    const std::string custom_prefix( _ui.lineEdit_prefix->text().toLocal8Bit());
    const std::string custom_export_path( _ui.lineEdit_path->text().toLocal8Bit());
    std::vector<std::string> export_files(_dicom_series_files.size());
    for (int i = 0 ; i<_dicom_series_files.size() ; ++i)
    {
        std::string file_input = _dicom_series_files[i];

        int postfix = -1;
        if (file_input.size() > 4 && 
            (file_input[file_input.size() - 1] == 'm' || file_input[file_input.size() - 1] == 'M') &&
            (file_input[file_input.size() - 2] == 'c' || file_input[file_input.size() - 2] == 'C') &&
            (file_input[file_input.size() - 3] == 'd' || file_input[file_input.size() - 3] == 'D') &&
            file_input[file_input.size() - 4] == '.')
        {
            postfix = file_input.size() - 4;
        }

        int sub = file_input.size() - 1;
        for (; sub>=0 ; --sub)
        {
            if (file_input[sub] == '\\' || file_input[sub] == '/')
            {
                break;
            }
        }

        std::string file_nake;
        if (postfix == -1)//no 
        {
            file_nake = file_input.substr(sub + 1 , file_input.size() -1 - sub);
        }
        else
        {
            file_nake = file_input.substr(sub + 1 , postfix - sub -1);
        }
        
        std::string file_name = custom_export_path + std::string("\\") + custom_prefix + file_nake + custom_postfix;
        export_files[i] = file_name;

        //std::cout << file_name << std::endl;
    }

    //3 If without private
    const ExportDicomDataType export_type = _ui.checkBox_remove_private_tags->isChecked() ?
        EXPORT_ANONYMOUS_DICOM_WITHOUT_PRIVATETAG : EXPORT_ANONYMOUS_DICOM;

    //4 Show progress dialog
    QApplication::setOverrideCursor(Qt::WaitCursor);

    std::shared_ptr<ProgressObserver> progress_ob(new ProgressObserver());
    QProgressDialog progress_dialog(tr("Exporting anonymous DICOM series ......") ,0 , 0 , 100 );

    _model_progress->clear_observer();
    _model_progress->add_observer(progress_ob);
    progress_ob->set_progress_model(_model_progress);
    progress_ob->set_progress_dialog(&progress_dialog);

    progress_dialog.setWindowTitle(tr("please wait."));
    progress_dialog.setFixedWidth(300);
    progress_dialog.setWindowModality(Qt::WindowModal);
    progress_dialog.show();

    //5 Export DICOM series
    DICOMExporter exporter;
    exporter.set_anonymous_taglist(anonymous_tags);
    exporter.set_progress_model(_model_progress);
    IOStatus status = exporter.export_series(_dicom_series_files , export_files , export_type);
    progress_dialog.close();

    if (status != IO_SUCCESS)
    {
        QApplication::restoreOverrideCursor();
        QMessageBox::warning(this , tr("Error") , tr("Export anonymous DICOM series failed!"));
        _model_progress->clear_observer();
    }
    else
    {
        QApplication::restoreOverrideCursor();
        QMessageBox::information(this , tr("Success") , tr("Export anonymous DICOM series success!"));
        _model_progress->clear_observer();
    }
}

void DICOMAnonymizationDlg::slot_press_btn_cancel_i()
{
    this->close();
}

void DICOMAnonymizationDlg::set_progress_model(std::shared_ptr<medical_imaging::ProgressModel> model)
{
    _model_progress = model;
}


