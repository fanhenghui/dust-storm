#include "mi_raw_data_import_dialog.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgIO/mi_zlib_utils.h"

#include <QFileDialog>

using namespace medical_imaging;

RawDataImportDlg::RawDataImportDlg(QWidget *parent /*= 0*/, Qt::WindowFlags f /*= 0*/)
{
    _ui.setupUi(this);

    //Initialize widget default value
    _ui.lineEdit_width->setValidator(new QIntValidator(1 , 3000 , _ui.lineEdit_width));
    _ui.lineEdit_height->setValidator(new QIntValidator(1 , 3000 , _ui.lineEdit_height));
    _ui.lineEdit_depth->setValidator(new QIntValidator(1 , 3000 , _ui.lineEdit_depth));

    _ui.lineEdit_width->setText("1");
    _ui.lineEdit_height->setText("1");
    _ui.lineEdit_depth->setText("1");

    _ui.lineEdit_origin_x->setValidator(new QDoubleValidator(_ui.lineEdit_origin_x));
    _ui.lineEdit_origin_y->setValidator(new QDoubleValidator(_ui.lineEdit_origin_y));
    _ui.lineEdit_origin_z->setValidator(new QDoubleValidator(_ui.lineEdit_origin_z));

    _ui.lineEdit_origin_x->setText("0.0");
    _ui.lineEdit_origin_y->setText("0.0");
    _ui.lineEdit_origin_z->setText("0.0");

    _ui.lineEdit_ori_x0->setValidator(new QDoubleValidator(_ui.lineEdit_ori_x0));
    _ui.lineEdit_ori_x1->setValidator(new QDoubleValidator(_ui.lineEdit_ori_x1));
    _ui.lineEdit_ori_x2->setValidator(new QDoubleValidator(_ui.lineEdit_ori_x2));

    _ui.lineEdit_ori_x0->setText("1.0");
    _ui.lineEdit_ori_x1->setText("0.0");
    _ui.lineEdit_ori_x2->setText("0.0");

    _ui.lineEdit_ori_y0->setValidator(new QDoubleValidator(_ui.lineEdit_ori_y0));
    _ui.lineEdit_ori_y1->setValidator(new QDoubleValidator(_ui.lineEdit_ori_y1));
    _ui.lineEdit_ori_y2->setValidator(new QDoubleValidator(_ui.lineEdit_ori_y2));

    _ui.lineEdit_ori_y0->setText("0.0");
    _ui.lineEdit_ori_y1->setText("1.0");
    _ui.lineEdit_ori_y2->setText("0.0");

    _ui.lineEdit_ori_z0->setValidator(new QDoubleValidator(_ui.lineEdit_ori_z0));
    _ui.lineEdit_ori_z1->setValidator(new QDoubleValidator(_ui.lineEdit_ori_z1));
    _ui.lineEdit_ori_z2->setValidator(new QDoubleValidator(_ui.lineEdit_ori_z2));

    _ui.lineEdit_ori_z0->setText("0.0");
    _ui.lineEdit_ori_z1->setText("0.0");
    _ui.lineEdit_ori_z2->setText("1.0");

    connect(_ui.pushButton_browse_path , SIGNAL(pressed()) , this , SLOT(slot_press_btn_browse_path_i()));
    connect(_ui.pushButton_cancel , SIGNAL(pressed()) , this , SLOT(slot_press_btn_cancel_i()));
    connect(_ui.pushButton_import , SIGNAL(pressed()) , this , SLOT(slot_press_btn_import_i()));

}

RawDataImportDlg::~RawDataImportDlg()
{

}

void RawDataImportDlg::slot_press_btn_browse_path_i()
{
    QString file_name = QFileDialog::getOpenFileName(this , tr("Import raw data") , "" , "raw data (*.raw *.zraw);;other(*)");
    _ui.lineEdit_path->setText(file_name);
}

void RawDataImportDlg::slot_press_btn_import_i()
{
    //IOStatus 
    //const std::string file_path(_ui.lineEdit_path->text().toLocal8Bit());
    //if (file_path.empty())
    //{
    //    
    //}
}

void RawDataImportDlg::slot_press_btn_cancel_i()
{
    this->close();
}
