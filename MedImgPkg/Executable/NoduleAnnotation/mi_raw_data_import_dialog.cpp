#include "mi_raw_data_import_dialog.h"

#include "MedImgUtil/mi_string_number_converter.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgIO/mi_zlib_utils.h"

#include "MedImgArithmetic/mi_vector3.h"
#include "MedImgArithmetic/mi_point3.h"

#include <QFileDialog>
#include <QMessageBox>

using namespace medical_imaging;

RawDataImportDlg::RawDataImportDlg(QWidget *parent /*= 0*/, Qt::WindowFlags f /*= 0*/)
{
    _ui.setupUi(this);

    this->setWindowFlags(Qt::Dialog | Qt::WindowCloseButtonHint);

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

    _ui.lineEdit_spacing0->setValidator(new QDoubleValidator(_ui.lineEdit_spacing0));
    _ui.lineEdit_spacing1->setValidator(new QDoubleValidator(_ui.lineEdit_spacing1));
    _ui.lineEdit_spacing2->setValidator(new QDoubleValidator(_ui.lineEdit_spacing2));

    _ui.lineEdit_spacing0->setText("0.0");
    _ui.lineEdit_spacing1->setText("0.0");
    _ui.lineEdit_spacing2->setText("1.0");

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
    IOStatus status;
    StrNumConverter<double> str_to_double;
    StrNumConverter<int> str_to_int;

    try
    {
        const std::string file_path(_ui.lineEdit_path->text().toLocal8Bit());
        if (file_path.empty())
        {
            status = IO_FILE_OPEN_FAILED;
            IO_THROW_EXCEPTION(str_to_int.to_string(status));
        }

        int dim[3] = {0,0,0};
        Point3 origin;
        Vector3 ori_x , ori_y , ori_z;
        double spacing[3] = {0,0,0};

        bool is_compressed = _ui.checkBox_compressed->isChecked();

        dim[0] = str_to_int.to_num(_ui.lineEdit_width->text().toStdString());
        dim[1] = str_to_int.to_num(_ui.lineEdit_height->text().toStdString());
        dim[2] = str_to_int.to_num(_ui.lineEdit_depth->text().toStdString());

        if (dim[0] < 1 || dim[1] <1 || dim[2] < 1)
        {
            status = IO_EMPTY_INPUT;
            IO_THROW_EXCEPTION(str_to_int.to_string(status));
        }

        spacing[0] = str_to_double.to_num(_ui.lineEdit_spacing0->text().toStdString());
        spacing[1] = str_to_double.to_num(_ui.lineEdit_spacing1->text().toStdString());
        spacing[2] = str_to_double.to_num(_ui.lineEdit_spacing2->text().toStdString());

        if (spacing[0] < DOUBLE_EPSILON || spacing[1] < DOUBLE_EPSILON || spacing[2] < DOUBLE_EPSILON)
        {
            status = IO_EMPTY_INPUT;
            IO_THROW_EXCEPTION(str_to_int.to_string(status));
        }

        ori_x.x = str_to_double.to_num(_ui.lineEdit_ori_x0->text().toStdString());
        ori_x.y = str_to_double.to_num(_ui.lineEdit_ori_x1->text().toStdString());
        ori_x.z = str_to_double.to_num(_ui.lineEdit_ori_x2->text().toStdString());

        ori_y.x = str_to_double.to_num(_ui.lineEdit_ori_y0->text().toStdString());
        ori_y.y = str_to_double.to_num(_ui.lineEdit_ori_y1->text().toStdString());
        ori_y.z = str_to_double.to_num(_ui.lineEdit_ori_y2->text().toStdString());

        ori_z.x = str_to_double.to_num(_ui.lineEdit_ori_z0->text().toStdString());
        ori_z.y = str_to_double.to_num(_ui.lineEdit_ori_z1->text().toStdString());
        ori_z.z = str_to_double.to_num(_ui.lineEdit_ori_z2->text().toStdString());

        origin.x = str_to_double.to_num(_ui.lineEdit_origin_x->text().toStdString());
        origin.y = str_to_double.to_num(_ui.lineEdit_origin_y->text().toStdString());
        origin.z = str_to_double.to_num(_ui.lineEdit_origin_z->text().toStdString());

        const std::string data_type_str = _ui.comboBox_data_type->currentText().toStdString();
        DataType data_type = USHORT;
        if (data_type_str == "ushort")
        {
            data_type = USHORT;
        }
        else if (data_type_str == "short")
        {
            data_type = SHORT;
        }
        else if (data_type_str == "uchar")
        {
            data_type = UCHAR;
        }
        else if (data_type_str == "char")
        {
            data_type = CHAR;
        }
        else if (data_type_str == "float")
        {
            data_type = FLOAT;
        }
        else
        {
            status = IO_UNSUPPORTED_YET;
            IO_THROW_EXCEPTION(str_to_int.to_string(status));
        }

        std::ifstream in(file_path , std::ios::in | std::ios::binary);
        if (!in.is_open())
        {
            status = IO_FILE_OPEN_FAILED;
            IO_THROW_EXCEPTION(str_to_int.to_string(status));
        }
        in.close();

        std::shared_ptr<ImageData> img_data(new ImageData());
        std::shared_ptr<ImageDataHeader> data_header(new ImageDataHeader());
        data_header->columns = dim[0];
        data_header->rows = dim[1];
        data_header->pixel_spacing[0] = spacing[0];
        data_header->pixel_spacing[1] = spacing[1];

        for (int i = 0 ; i<3 ; ++i)
        {
            img_data->_dim[i] = static_cast<unsigned int>(dim[i]);
            img_data->_spacing[i] = spacing[i];
        }

        img_data->_data_type = data_type;
        img_data->_image_position = origin;
        img_data->_image_orientation[0] = ori_x;
        img_data->_image_orientation[1] = ori_y;
        img_data->_image_orientation[2] = ori_z;
        img_data->_channel_num =  1;
        img_data->mem_allocate();

        if (is_compressed)
        {
            int out_size(0);
            status = ZLibUtils::decompress(file_path , (char*)img_data->get_pixel_pointer() , out_size);
            if (status != IO_SUCCESS)
            {
                IO_THROW_EXCEPTION(str_to_int.to_string(status));
            }
        }
        else
        {
            in.open(file_path , std::ios::binary | std::ios::in);
            in.read((char*)img_data->get_pixel_pointer() , img_data->get_data_size());
            if (in.gcount() != img_data->get_data_size())
            {
                status = IO_DATA_DAMAGE;
                in.close();
                IO_THROW_EXCEPTION(str_to_int.to_string(status));
            }
            in.close();
        }

        emit raw_data_imported(img_data , data_header);

        this->close();

    }
    catch (const Exception& e)
    {
        QMessageBox::warning(this , tr("Error") , tr("Import raw data failed!"));
        this->close();
    }
}

void RawDataImportDlg::slot_press_btn_cancel_i()
{
    this->close();
}
