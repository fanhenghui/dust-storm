#include "my_window.h"

#include <iostream>

#include "MedImgCommon/mi_concurrency.h"
#include "MedImgCommon/mi_configuration.h"
#include "MedImgCommon/mi_string_number_converter.h"

#include "MedImgArithmetic/mi_rsa_utils.h"
#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgIO/mi_meta_object_loader.h"
#include "MedImgIO/mi_nodule_set.h"
#include "MedImgIO/mi_nodule_set_parser.h"
#include "MedImgIO/mi_model_progress.h"

#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

#include "MedImgQtWidgets/mi_shared_widget.h"

#include "my_graphic_scene.h"
#include "my_graphic_view.h"
#include "my_graphic_items.h"

#include <QEvent>
#include <QSizePolicy>
#include <QScrollBar>
#include <QFileDialog>
#include <QMessagebox>
#include <QSignalMapper>
#include <QProgressDialog>

using namespace medical_imaging;

MyWindow::MyWindow(QWidget *parent /*= 0*/):
        QMainWindow(parent)
{
    ui.setupUi(this);

    //this->setCentralWidget(_my_graphic_view[i]);
    this->setWindowTitle("Graphic view");

    for (int i = 0  ; i< 4; ++i)
    {
        //_my_graphic_scene[i] = new MyGraphicScene(this);
        _my_graphic_view[i] = new MyGraphicView(this);
        _my_graphic_view[i]->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);//×ÔÊÊÓ¦´°¿Ú
        _my_graphic_view[i]->setSceneRect(0,0 , _my_graphic_view[i]->width() , _my_graphic_view[i]->height());

        std::cout << _my_graphic_view[i]->width()/2 << " " << _my_graphic_view[i]->height()/2 << std::endl;
        //text item
        MyTextItem* text_item = new MyTextItem();
        text_item->setPlainText(tr("Hello world \nNull"));
        text_item->setPos(_my_graphic_view[i]->width()/2 , _my_graphic_view[i]->height()/2);
        _my_graphic_view[i]->get_graphics_scene()->addItem(text_item);

        //ellipse
        MyEllipse* ellipse_item  = new MyEllipse();
        ellipse_item->setRect(0,0,50,50);
        ellipse_item->setPos(0,0);
        _my_graphic_view[i]->get_graphics_scene()->addItem(ellipse_item);

        MyEllipse* ellipse_item2  = new MyEllipse();
        ellipse_item2->setRect(0,0,50,50);
        ellipse_item2->setPos(25,25);
        //ellipse_item2->hide();
        _my_graphic_view[i]->get_graphics_scene()->addItem(ellipse_item2);
    }

    ui.gridLayout->addWidget(_my_graphic_view[0] , 0 , 0);
    ui.gridLayout->addWidget(_my_graphic_view[1] , 0 , 1);
    ui.gridLayout->addWidget(_my_graphic_view[2] , 1 , 0);
    ui.gridLayout->addWidget(_my_graphic_view[3] , 1 , 1);

    connect(ui.actionOpen_DICOM_folder , SIGNAL(triggered()) , this , SLOT(slot_open_dicom_folder()));

    configure();
}

MyWindow::~MyWindow()
{

}

void MyWindow::slot_open_dicom_folder()
{
    QStringList file_name_list = QFileDialog::getOpenFileNames(
        this ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    if (!file_name_list.empty())
    {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        std::vector<std::string> file_names_std(file_name_list.size());
        int idx = 0;
        for (auto it = file_name_list.begin() ; it != file_name_list.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            file_names_std[idx++] = s;
        }

        std::shared_ptr<ImageDataHeader> data_header;
        std::shared_ptr<ImageData> image_data;

        DICOMLoader loader;
        IOStatus status = loader.load_series(file_names_std, image_data , data_header);
        if (status != IO_SUCCESS)
        {
            QApplication::restoreOverrideCursor();
            QMessageBox::warning(this , tr("load DICOM Folder") , tr("load DICOM folder failed!"));
            return;
        }

        SharedWidget::instance();
        if (_volume_infos)//Delete last one
        {
            _volume_infos->finialize();
        }
        _volume_infos.reset(new VolumeInfos());
        _volume_infos->set_data_header(data_header);
        //SharedWidget::instance()->makeCurrent();
        _volume_infos->set_volume(image_data);//load volume texture if has graphic card

        //create_model_observer_i();

        create_scenes();

        QApplication::restoreOverrideCursor();

        //_mpr_00->update();
        //_mpr_01->update();
        //_mpr10->update();

        //_model_progress->clear_observer();

        //reset nodule list
        //ui.tableWidgetNoduleList->clear();
        //ui.tableWidgetNoduleList->setRowCount(0);
        //_select_vio_id = -1;
    }
    else
    {
        return;
    }
}

void MyWindow::create_scenes()
{
    const float PRESET_CT_LUNGS_WW = 1500;
    const float PRESET_CT_LUNGS_WL = -400;

    ScanSliceType scan_types[3] = {SAGITTAL ,CORONAL , TRANSVERSE};
    const std::string scan_types_string[3] = {"Sagittal_MPR_scene_00" ,"Coronal_MPR_scene_01" , "Transverse_MPR_scene_10"};

    for (int i = 0; i< 3 ; ++i)
    {
        _mpr_scenes[i].reset(new MPRScene(_my_graphic_view[i]->width() , _my_graphic_view[i]->height()));

        _mpr_scenes[i]->set_volume_infos(_volume_infos);
        _mpr_scenes[i]->set_sample_rate(1.0);
        _mpr_scenes[i]->set_global_window_level(PRESET_CT_LUNGS_WW,PRESET_CT_LUNGS_WL);
        _mpr_scenes[i]->set_composite_mode(COMPOSITE_AVERAGE);
        _mpr_scenes[i]->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
        _mpr_scenes[i]->set_mask_mode(MASK_NONE);
        _mpr_scenes[i]->set_interpolation_mode(LINEAR);

        _mpr_scenes[i]->place_mpr(scan_types[i]);
        _mpr_scenes[i]->set_name(scan_types_string[i]);

        _my_graphic_view[i]->set_scene(_mpr_scenes[i]);
    }

    for (int i= 0 ; i< 3 ; ++i)
    {
        _my_graphic_view[i]->update_scene();
    }
}

void MyWindow::configure()
{
    Configuration::instance()->set_processing_unit_type(GPU);
    Configuration::instance()->set_nodule_file_rsa(true);

    GLUtils::set_check_gl_flag(false);
}
