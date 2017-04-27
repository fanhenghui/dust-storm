#include "mi_main_window.h"

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
#include "MedImgQtWidgets/mi_scene_container.h"
#include "MedImgQtWidgets/mi_graphic_item_corners_info.h"
#include "MedImgQtWidgets/mi_graphic_item_voi.h"
#include "MedImgQtWidgets/mi_graphic_item_cross_hair.h"
#include "MedImgQtWidgets/mi_graphic_item_mpr_border.h"
#include "MedImgQtWidgets/mi_mouse_op_zoom.h"
#include "MedImgQtWidgets/mi_mouse_op_pan.h"
#include "MedImgQtWidgets/mi_mouse_op_rotate.h"
#include "MedImgQtWidgets/mi_mouse_op_mpr_page.h"
#include "MedImgQtWidgets/mi_mouse_op_windowing.h"
#include "MedImgQtWidgets/mi_mouse_op_probe.h"
#include "MedImgQtWidgets/mi_mouse_op_annotate.h"
#include "MedImgQtWidgets/mi_mouse_op_locate.h"
#include "MedImgQtWidgets/mi_model_voi.h"
#include "MedImgQtWidgets/mi_model_cross_hair.h"
#include "MedImgQtWidgets/mi_observer_scene_container.h"
#include "MedImgQtWidgets/mi_observer_progress.h"
#include "MedImgQtWidgets/mi_observer_voi_statistic.h"


#include "mi_observer_voi_table.h"
#include "mi_observer_mpr_scroll_bar.h"
#include "mi_mouse_op_min_max_hint.h"
#include "mi_my_rsa.h"

#include <QEvent>
#include <QSizePolicy>
#include <QScrollBar>
#include <QFileDialog>
#include <QMessagebox>
#include <QSignalMapper>
#include <QProgressDialog>

using namespace medical_imaging;

//Nodule type
const std::string NODULE_TYPE_GGN = std::string("GGN");
const std::string NODULE_TYPE_AAH = std::string("AAH");

//Preset WL
const float PRESET_CT_ABDOMEN_WW = 400;
const float PRESET_CT_ABDOMEN_WL = 60;

const float PRESET_CT_LUNGS_WW = 1500;
const float PRESET_CT_LUNGS_WL = -400;

const float PRESET_CT_BRAIN_WW = 80;
const float PRESET_CT_BRAIN_WL = 40;

const float PRESET_CT_ANGIO_WW = 600;
const float PRESET_CT_ANGIO_WL = 300;

const float PRESET_CT_BONE_WW = 1500;
const float PRESET_CT_BONE_WL = 300;

const float PRESET_CT_CHEST_WW = 400;
const float PRESET_CT_CHEST_WL = 40;

NoduleAnnotation::NoduleAnnotation(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags),
    _layout_tag(0),
    _is_ready(false),
    _object_nodule(nullptr),
    _single_manager_nodule_type(nullptr),
    _select_vio_id(-1)
{
    ui.setupUi(this);

    ui.tableWidgetNoduleList->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui.tableWidgetNoduleList->setSelectionMode(QAbstractItemView::SingleSelection);

    _mpr_00 = new SceneContainer(SharedWidget::instance());
    _mpr_01 = new SceneContainer(SharedWidget::instance());
    _mpr10 = new SceneContainer(SharedWidget::instance());
    _vr_11 = new SceneContainer(SharedWidget::instance());

    _mpr10->setMinimumSize(100,100);
    _mpr_01->setMinimumSize(100,100);
    _mpr_00->setMinimumSize(100,100);
    _vr_11->setMinimumSize(100,100);

    //_mpr_00->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);//自适应窗口
    //_mpr_01->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);
    //_mpr10->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);
    //_vr_11->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);


    _mpr_00_scroll_bar = new QScrollBar(ui.centralWidget);
    _mpr_00_scroll_bar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR00"));
    _mpr_00_scroll_bar->setOrientation(Qt::Vertical);

    _mpr_01_scroll_bar = new QScrollBar(ui.centralWidget);
    _mpr_01_scroll_bar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR01"));
    _mpr_01_scroll_bar->setOrientation(Qt::Vertical);

    _mpr_10_scroll_bar = new QScrollBar(ui.centralWidget);
    _mpr_10_scroll_bar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR10"));
    _mpr_10_scroll_bar->setOrientation(Qt::Vertical);


    ui.gridLayout_6->addWidget(_mpr_00 , 0 ,0);
    ui.gridLayout_6->addWidget(_mpr_00_scroll_bar , 0 ,1,1,1);
    ui.gridLayout_6->addWidget(_mpr_01 , 0 ,2);
    ui.gridLayout_6->addWidget(_mpr_01_scroll_bar , 0 ,3,1,1);
    ui.gridLayout_6->addWidget(_mpr10 , 1 ,0);
    ui.gridLayout_6->addWidget(_mpr_10_scroll_bar , 1 ,1,1,1);
    ui.gridLayout_6->addWidget(_vr_11 , 1 ,2);

    _object_nodule = new QNoduleObject(this);
    _object_min_max_hint = new QMinMaxHintObject(this);

    //progress model
    _model_progress.reset( new ProgressModel());

    configure_i();

    connect_signal_slot_i();
}

NoduleAnnotation::~NoduleAnnotation()
{

}

void NoduleAnnotation::configure_i()
{
    //1 TODO Check process unit
    Configuration::instance()->set_processing_unit_type(GPU);
    Configuration::instance()->set_nodule_file_rsa(true);

    GLUtils::set_check_gl_flag(false);
}

void NoduleAnnotation::create_scene_i()
{
    _mpr_scene_00.reset(new MPRScene(_mpr_00->width() , _mpr_00->height()));
    _mpr_scene_01.reset(new MPRScene(_mpr_01->width() , _mpr_01->height()));
    _mpr_scene_10.reset(new MPRScene(_mpr10->width() , _mpr10->height()));

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        //1 Set Scene
        mpr_containers[i]->set_scene(mpr_scenes[i]);

        //2 Set scene parameter
        mpr_scenes[i]->set_volume_infos(_volume_infos);
        mpr_scenes[i]->set_sample_rate(1.0);
        mpr_scenes[i]->set_global_window_level(PRESET_CT_LUNGS_WW,PRESET_CT_LUNGS_WL);
        mpr_scenes[i]->set_composite_mode(COMPOSITE_AVERAGE);
        mpr_scenes[i]->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
        mpr_scenes[i]->set_mask_mode(MASK_NONE);
        mpr_scenes[i]->set_interpolation_mode(LINEAR);

        //3 Add painter list
        std::shared_ptr<GraphicItemCornersInfo> graphic_item_corner_info(new GraphicItemCornersInfo());
        graphic_item_corner_info->set_scene(mpr_scenes[i]);
        mpr_containers[i]->add_item(graphic_item_corner_info);

        std::shared_ptr<GraphicItemVOI> graphic_item_voi(new GraphicItemVOI());
        graphic_item_voi->set_scene(mpr_scenes[i]);
        graphic_item_voi->set_voi_model(_model_voi);
        mpr_containers[i]->add_item(graphic_item_voi);

        std::shared_ptr<GraphicItemCrosshair> graphic_item_crosshair(new GraphicItemCrosshair());
        graphic_item_crosshair->set_scene(mpr_scenes[i]);
        graphic_item_crosshair->set_crosshair_model(_model_crosshair);
        mpr_containers[i]->add_item(graphic_item_crosshair);

        std::shared_ptr<GraphicItemMPRBorder> graphic_item_mpr_border(new GraphicItemMPRBorder());
        graphic_item_mpr_border->set_scene(mpr_scenes[i]);
        graphic_item_mpr_border->set_crosshair_model(_model_crosshair);
        mpr_containers[i]->add_item(graphic_item_mpr_border);

        //4 Add operation 
        std::shared_ptr<MouseOpLocate> op_mpr_locate(new MouseOpLocate());
        op_mpr_locate->set_scene(mpr_scenes[i]);
        op_mpr_locate->set_crosshair_model(_model_crosshair);

        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);

        IMouseOpPtrCollection left_btn_ops(2);
        left_btn_ops[0] = op_mpr_locate;
        left_btn_ops[1] = op_min_max_hint;
        mpr_containers[i]->register_mouse_operation(left_btn_ops, Qt::LeftButton , Qt::NoModifier);

        std::shared_ptr<MouseOpZoom> op_zoom(new MouseOpZoom());
        op_zoom->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_zoom , Qt::RightButton , Qt::NoModifier);

        std::shared_ptr<MouseOpWindowing> op_windowing(new MouseOpWindowing());
        op_windowing->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_windowing , Qt::MiddleButton , Qt::NoModifier);

        std::shared_ptr<MouseOpPan> op_pan(new MouseOpPan());
        op_pan->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_pan , Qt::LeftButton | Qt::RightButton , Qt::NoModifier);

        std::shared_ptr<MouseOpMPRPage> op_page(new MouseOpMPRPage());
        op_page->set_scene(mpr_scenes[i]);
        op_page->set_crosshair_model(_model_crosshair);
        mpr_containers[i]->register_mouse_wheel_operation(op_page);

    }

    //////////////////////////////////////////////////////////////////////////
    //Placement orthogonal MPR
    std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();
    ScanSliceType scan_types[3] = {SAGITTAL ,CORONAL , TRANSVERSE};
    const std::string scan_types_string[3] = {"Sagittal_MPR_scene_00" ,"Coronal_MPR_scene_01" , "Transverse_MPR_scene_10"};
    MPRScenePtr scenes[3] = {_mpr_scene_00 , _mpr_scene_01 , _mpr_scene_10};
    RGBUnit colors[3] ={kColorSagittal, kColorCoronal , kColorTransverse};
    QScrollBar* scroll_bars[3] = {_mpr_00_scroll_bar , _mpr_01_scroll_bar , _mpr_10_scroll_bar};

    //Model set scenes
    _model_crosshair->set_mpr_scene(scan_types , scenes , colors);

    _ob_mpr_scroll_bar->add_scroll_bar(_mpr_scene_00 , _mpr_00_scroll_bar);
    _ob_mpr_scroll_bar->add_scroll_bar(_mpr_scene_01 , _mpr_01_scroll_bar);
    _ob_mpr_scroll_bar->add_scroll_bar(_mpr_scene_10 , _mpr_10_scroll_bar);

    for (int i = 0 ; i<3 ; ++i)
    {
        scenes[i]->place_mpr(scan_types[i]);
        scenes[i]->set_name(scan_types_string[i]);

        //Init page
        scroll_bars[i]->setMaximum(camera_cal->get_page_maximum(scan_types[i])-1);
        scroll_bars[i]->setMinimum(0);
        scroll_bars[i]->setPageStep(1);
        scroll_bars[i]->setValue(camera_cal->get_default_page(scan_types[i]));
    }

    //////////////////////////////////////////////////////////////////////////
    //focus in/out scene signal mapper
    QSignalMapper* focus_in_singal_mapper = new QSignalMapper();

    connect(_mpr_00 , SIGNAL(focus_in_scene()) , focus_in_singal_mapper , SLOT(map()));
    focus_in_singal_mapper->setMapping(_mpr_00 , QString(_mpr_00->get_name().c_str()));
    connect(_mpr_01 , SIGNAL(focus_in_scene()) , focus_in_singal_mapper , SLOT(map()));
    focus_in_singal_mapper->setMapping(_mpr_01 , QString(_mpr_01->get_name().c_str()));
    connect(_mpr10 , SIGNAL(focus_in_scene()) , focus_in_singal_mapper , SLOT(map()));
    focus_in_singal_mapper->setMapping(_mpr10 , QString(_mpr10->get_name().c_str()));

    connect(focus_in_singal_mapper , SIGNAL(mapped(QString)) , this , SLOT(slot_focus_in_scene_i(QString)));

    QSignalMapper* focus_out_singal_mapper = new QSignalMapper();

    connect(_mpr_00 , SIGNAL(focus_out_scene()) , focus_out_singal_mapper , SLOT(map()));
    focus_out_singal_mapper->setMapping(_mpr_00 , QString(_mpr_00->get_name().c_str()));
    connect(_mpr_01 , SIGNAL(focus_out_scene()) , focus_out_singal_mapper , SLOT(map()));
    focus_out_singal_mapper->setMapping(_mpr_01 , QString(_mpr_01->get_name().c_str()));
    connect(_mpr10 , SIGNAL(focus_out_scene()) , focus_out_singal_mapper , SLOT(map()));
    focus_out_singal_mapper->setMapping(_mpr10 , QString(_mpr10->get_name().c_str()));

    connect(focus_out_singal_mapper , SIGNAL(mapped(QString)) , this , SLOT(slot_focus_out_scene_i(QString)));
    //////////////////////////////////////////////////////////////////////////
}

void NoduleAnnotation::connect_signal_slot_i()
{
    //Layout
    //connect(ui.action1x1 , SIGNAL(triggered()) , this , SLOT(SlotChangeLayout1x1_i()));
    connect(ui.action2x2 , SIGNAL(triggered()) , this , SLOT(slot_change_layout2x2_i()));

    //File
    connect(ui.actionOpen_DICOM_Folder , SIGNAL(triggered()) , this , SLOT(slot_open_dicom_folder_i()));
    connect(ui.actionOpen_Meta_Image , SIGNAL(triggered()) , this , SLOT(slot_open_meta_image_i()));
    connect(ui.actionOpen_Raw , SIGNAL(triggered()) , this , SLOT(slot_open_raw_i()));
    connect(ui.actionSave_Nodule , SIGNAL(triggered()) , this , SLOT(slot_save_nodule_file_i()));
    connect(ui.actionLoad_Nodule , SIGNAL(triggered()) , this , SLOT(slot_open_nodule_file_i()));

    //MPR scroll bar
    connect(_mpr_00_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr00_i(int)));
    connect(_mpr_01_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr01_i(int)));
    connect(_mpr_10_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr10_i(int)));

    //Common tools
    connect(ui.pushButtonArrow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_arrow_i()));
    connect(ui.pushButtonAnnotate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_annotate_i()));
    connect(ui.pushButtonRotate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_rotate_i()));
    connect(ui.pushButtonZoom , SIGNAL(pressed()) , this , SLOT(slot_press_btn_zoom_i()));
    connect(ui.pushButtonPan , SIGNAL(pressed()) , this , SLOT(slot_press_btn_pan_i()));
    connect(ui.pushButtonWindowing , SIGNAL(pressed()) , this , SLOT(slot_press_btn_windowing_i()));
    connect(ui.pushButtonFitWindow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_fit_window_i()));

    //VOI list
    connect(ui.tableWidgetNoduleList , SIGNAL(cellPressed(int,int)) , this , SLOT(slot_voi_table_widget_cell_select_i(int ,int)));
    connect(ui.tableWidgetNoduleList , SIGNAL(itemChanged(QTableWidgetItem *)) , this , SLOT(slot_voi_table_widget_item_changed_i(QTableWidgetItem *)));
    connect(_object_nodule , SIGNAL(nodule_added()) , this , SLOT(slot_add_nodule_i()));
    connect(ui.pushButtonDeleteNodule , SIGNAL(pressed()) , this , SLOT(slot_delete_nodule_i()));

    //Preset WL
    connect(ui.comboBoxPresetWL , SIGNAL(currentIndexChanged(QString)) , this , SLOT(slot_preset_wl_changed_i(QString)));

    //Scene Min Max hint
    connect(_object_min_max_hint , SIGNAL(triggered(const std::string&)) , this , SLOT(slot_scene_min_max_hint_i(const std::string&)));

    //Crosshair visibility
    connect(ui.checkBoxCrossHair , SIGNAL(stateChanged(int)) , this , SLOT(slot_crosshair_visibility_i(int)));
}

void NoduleAnnotation::create_model_observer_i()
{
    //VOI
    _model_voi.reset(new VOIModel());

    _ob_voi_table.reset(new VOITableObserver());
    _ob_voi_table->set_nodule_object(_object_nodule);

    _ob_scene_container.reset(new SceneContainerObserver());//万能 observer
    _ob_scene_container->add_scene_container(_mpr_00);
    _ob_scene_container->add_scene_container(_mpr_01);
    _ob_scene_container->add_scene_container(_mpr10);

    _ob_voi_statistic.reset(new VOIStatisticObserver());
    _ob_voi_statistic->set_model(_model_voi);
    _ob_voi_statistic->set_volume_infos(_volume_infos);

    _model_voi->add_observer(_ob_voi_statistic);
    _model_voi->add_observer(_ob_voi_table);

    //m_painter_voiModel->add_observer(m_pSceneContainerOb);//Scene的刷新通过change item来完成

    //Crosshair & cross location
    _model_crosshair.reset(new CrosshairModel());

    _ob_mpr_scroll_bar.reset(new MPRScrollBarObserver());
    _ob_mpr_scroll_bar->set_crosshair_model(_model_crosshair);

    _model_crosshair->add_observer(_ob_mpr_scroll_bar);
    _model_crosshair->add_observer(_ob_scene_container);

    if (!_single_manager_nodule_type)
    {
        delete _single_manager_nodule_type;
        _single_manager_nodule_type = new QSignalMapper(this);
        connect(_single_manager_nodule_type , SIGNAL(mapped(int)) , this , SLOT(slot_voi_table_widget_nodule_type_changed_i(int)));
    }
}

void NoduleAnnotation::slot_change_layout2x2_i()
{
    if (!_is_ready)
    {
        return;
    }

    _mpr_00->hide();
    _mpr_00_scroll_bar->hide();
    _mpr_01->hide();
    _mpr_01_scroll_bar->hide();
    _mpr10->hide();
    _mpr_10_scroll_bar->hide();
    _vr_11->hide();

    ui.gridLayout_6->removeWidget(_mpr_00);
    ui.gridLayout_6->removeWidget(_mpr_00_scroll_bar);
    ui.gridLayout_6->removeWidget(_mpr_01);
    ui.gridLayout_6->removeWidget(_mpr_01_scroll_bar);
    ui.gridLayout_6->removeWidget(_mpr10 );
    ui.gridLayout_6->removeWidget(_mpr_10_scroll_bar);
    ui.gridLayout_6->removeWidget(_vr_11);

    ui.gridLayout_6->addWidget(_mpr_00 , 0 ,0);
    ui.gridLayout_6->addWidget(_mpr_00_scroll_bar , 0 ,1,1,1);
    ui.gridLayout_6->addWidget(_mpr_01 , 0 ,2);
    ui.gridLayout_6->addWidget(_mpr_01_scroll_bar , 0 ,3,1,1);
    ui.gridLayout_6->addWidget(_mpr10 , 1 ,0);
    ui.gridLayout_6->addWidget(_mpr_10_scroll_bar , 1 ,1,1,1);
    ui.gridLayout_6->addWidget(_vr_11 , 1 ,2);

    //Set min size to fix size bug
    _mpr_00->setMinimumSize(_pre_2x2_width , _pre_2x2_height);
    _mpr_01->setMinimumSize(_pre_2x2_width , _pre_2x2_height);
    _mpr10->setMinimumSize(_pre_2x2_width , _pre_2x2_height);
    _vr_11->setMinimumSize(_pre_2x2_width ,_pre_2x2_height);
    

    _mpr_00->show();
    _mpr_00_scroll_bar->show();
    _mpr_01->show();
    _mpr_01_scroll_bar->show();
    _mpr10->show();
    _mpr_10_scroll_bar->show();
    _vr_11->show();

    //Recover min size to expanding
    _mpr10->setMinimumSize(100,100);
    _mpr_01->setMinimumSize(100,100);
    _mpr_00->setMinimumSize(100,100);
    _vr_11->setMinimumSize(100,100);

    _layout_tag = 0;
}

void NoduleAnnotation::slot_open_dicom_folder_i()
{
    QStringList file_name_list = QFileDialog::getOpenFileNames(
        this ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    if (!file_name_list.empty())
    {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        //Init progress dialog
        DICOMLoader loader;
        _model_progress->clear_observer();
        std::shared_ptr<ProgressObserver> progress_ob(new ProgressObserver());
        progress_ob->set_progress_model(_model_progress);
        _model_progress->add_observer(progress_ob);
        loader.set_progress_model(_model_progress);

        QProgressDialog progress_dialog(tr("Loading DICOM series ......") ,0 , 0 , 100 , this );
        progress_dialog.setWindowTitle(tr("please wait."));
        progress_dialog.setFixedWidth(300);
        progress_dialog.setWindowModality(Qt::WindowModal);
        progress_ob->set_progress_dialog(&progress_dialog);
        progress_dialog.show();

        _model_progress->set_progress(0);
        _model_progress->notify();

        std::vector<std::string> file_names_std(file_name_list.size());
        int idx = 0;
        for (auto it = file_name_list.begin() ; it != file_name_list.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            file_names_std[idx++] = s;
        }

        std::shared_ptr<ImageDataHeader> data_header;
        std::shared_ptr<ImageData> image_data;
        IOStatus status = loader.load_series(file_names_std, image_data , data_header);
        if (status != IO_SUCCESS)
        {
            QApplication::restoreOverrideCursor();
            QMessageBox::warning(this , tr("load DICOM Folder") , tr("load DICOM folder failed!"));
            _model_progress->clear_observer();
            return;
        }

        if (_volume_infos)//Delete last one
        {
            _volume_infos->finialize();
        }
        _volume_infos.reset(new VolumeInfos());
        _volume_infos->set_data_header(data_header);
        //SharedWidget::instance()->makeCurrent();
        _volume_infos->set_volume(image_data);//load volume texture if has graphic card

        create_model_observer_i();

        create_scene_i();

        save_layout2x2_parameter_i();

        QApplication::restoreOverrideCursor();

        _mpr_00->update();
        _mpr_01->update();
        _mpr10->update();

        _model_progress->clear_observer();

        //reset nodule list
        ui.tableWidgetNoduleList->clear();
        ui.tableWidgetNoduleList->setRowCount(0);
        _select_vio_id = -1;

        _is_ready = true;
    }
    else
    {
        return;
    }
}

void NoduleAnnotation::slot_open_meta_image_i()
{

}

void NoduleAnnotation::slot_open_raw_i()
{

}

void NoduleAnnotation::slot_press_btn_annotate_i()
{
    if (!_is_ready)
    {
        return;
    }

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpAnnotate> op_annotate(new MouseOpAnnotate());
        op_annotate->set_scene(mpr_scenes[i]);
        op_annotate->set_voi_model(_model_voi);//Set Model to annotate tools

        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);

        IMouseOpPtrCollection left_btn_ops(2);
        left_btn_ops[0] = op_annotate;
        left_btn_ops[1] = op_min_max_hint;

        mpr_containers[i]->register_mouse_operation(left_btn_ops , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_arrow_i()
{
    if (!_is_ready)
    {
        return;
    }

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpLocate> op_mpr_locate(new MouseOpLocate());
        op_mpr_locate->set_scene(mpr_scenes[i]);
        op_mpr_locate->set_crosshair_model(_model_crosshair);

        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);

        IMouseOpPtrCollection left_btn_ops(2);
        left_btn_ops[0] = op_mpr_locate;
        left_btn_ops[1] = op_min_max_hint;

        mpr_containers[i]->register_mouse_operation(left_btn_ops , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_rotate_i()
{
    if (!_is_ready)
    {
        return;
    }

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpRotate> op_rotate(new MouseOpRotate());
        op_rotate->set_scene(mpr_scenes[i]);

        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);

        IMouseOpPtrCollection left_btn_ops(2);
        left_btn_ops[0] = op_rotate;
        left_btn_ops[1] = op_min_max_hint;

        mpr_containers[i]->register_mouse_operation(left_btn_ops , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_zoom_i()
{
    if (!_is_ready)
    {
        return;
    }

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpZoom> op_zoom(new MouseOpZoom());
        op_zoom->set_scene(mpr_scenes[i]);

        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);

        IMouseOpPtrCollection left_btn_ops(2);
        left_btn_ops[0] = op_zoom;
        left_btn_ops[1] = op_min_max_hint;

        mpr_containers[i]->register_mouse_operation(left_btn_ops , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_pan_i()
{
    if (!_is_ready)
    {
        return;
    }

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpPan> op_pan(new MouseOpPan());
        op_pan->set_scene(mpr_scenes[i]);

        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);

        IMouseOpPtrCollection left_btn_ops(2);
        left_btn_ops[0] = op_pan;
        left_btn_ops[1] = op_min_max_hint;

        mpr_containers[i]->register_mouse_operation(left_btn_ops , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_windowing_i()
{
    if (!_is_ready)
    {
        return;
    }

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpWindowing> op_windowing(new MouseOpWindowing());
        op_windowing->set_scene(mpr_scenes[i]);

        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);

        IMouseOpPtrCollection left_btn_ops(2);
        left_btn_ops[0] = op_windowing;
        left_btn_ops[1] = op_min_max_hint;

        mpr_containers[i]->register_mouse_operation(left_btn_ops , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_fit_window_i()
{
    //TODO
    if (!_is_ready)
    {
        return;
    }

    if (_mpr_00->hasFocus())
    {
        _mpr_scene_00->place_mpr(SAGITTAL);
        _model_crosshair->set_changed();
        _model_voi->set_changed();
    }
    else if (_mpr_01->hasFocus())
    {
        _mpr_scene_01->place_mpr(CORONAL);
        _model_crosshair->set_changed();
        _model_voi->set_changed();
    }
    else if (_mpr10->hasFocus())
    {
        _mpr_scene_10->place_mpr(TRANSVERSE);
        _model_crosshair->set_changed();
        _model_voi->set_changed();
    }

    _model_crosshair->notify();
    _model_voi->notify();
}

void NoduleAnnotation::slot_save_nodule_file_i()
{
    if (!_is_ready)
    {
        return;
    }

    if (_model_voi->get_voi_spheres().empty())
    {
        if(QMessageBox::No == QMessageBox::warning(
            this , tr("Save Nodule") , tr("Nodule count is zero. If you still want to save to file?"),QMessageBox::Yes |QMessageBox::No))
        {
            return;
        }
    }

    QString file_name = QFileDialog::getSaveFileName(this, tr("Save Nodule") , QString(_volume_infos->get_data_header()->series_uid.c_str()), tr("NoduleSet(*.csv)"));
    if (!file_name.isEmpty())
    {
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        const std::vector<VOISphere>& vois = _model_voi->get_voi_spheres();
        nodule_set->set_nodule(vois);

        NoduleSetParser parser;
        std::string file_name_std(file_name.toLocal8Bit());

        IOStatus status;
        if (Configuration::instance()->get_nodule_file_rsa())
        {
            RSAUtils rsa_utils;
            mbedtls_rsa_context rsa;
            if(rsa_utils.to_rsa_context(S_N , S_E , S_D , S_P , S_Q , S_DP , S_DQ , S_QP , rsa) != 0)
            {
                status = IO_ENCRYPT_FAILED;
            }
            else
            {
                status = parser.save_as_rsa_binary(file_name_std , rsa , nodule_set);
            }
        }
        else
        {
            status = parser.save_as_csv(file_name_std , nodule_set);
        }
        

        if (status == IO_SUCCESS)
        {
            QMessageBox::information(this , tr("Save Nodule") , tr("Save nodule file success."),QMessageBox::Ok);
        }
        else
        {
            QMessageBox::warning(this , tr("Save Nodule") , tr("Save nodule file failed."),QMessageBox::Ok);
        }
        //TODO check status
    }
}

void NoduleAnnotation::slot_open_nodule_file_i()
{
    if (!_is_ready)
    {
        return;
    }

    if (!_model_voi->get_voi_spheres().empty())
    {
        if (QMessageBox::No == QMessageBox::warning(
            this , tr("Load Nodule") , tr("You had annotated some of nodule . Will you discard them and load a new nodule file"),
            QMessageBox::Yes | QMessageBox::No))
        {
            return;
        }
    }

    QString file_name = QFileDialog::getOpenFileName(this, tr("Load Nodule") , QString(_volume_infos->get_data_header()->series_uid.c_str()), tr("NoduleSet(*.csv)"));
    if (!file_name.isEmpty())
    {
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        NoduleSetParser parser;
        std::string file_name_std(file_name.toLocal8Bit());

        IOStatus status ;
        if (Configuration::instance()->get_nodule_file_rsa())
        {
            RSAUtils rsa_utils;
            mbedtls_rsa_context rsa;
            if(rsa_utils.to_rsa_context(S_N , S_E , S_D , S_P , S_Q , S_DP , S_DQ , S_QP , rsa) != 0)
            {
                status = IO_ENCRYPT_FAILED;
            }
            else
            {
                status = parser.load_as_rsa_binary(file_name_std , rsa , nodule_set);
            }
        }
        else
        {
            status = parser.load_as_csv(file_name_std , nodule_set);
        }

        if (status == IO_SUCCESS)
        {
            _model_voi->remove_all();
            const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
            for (auto it = vois.begin() ; it != vois.end() ; ++it)
            {
                _model_voi->add_voi_sphere(*it);
            }
            _model_voi->notify();
            QMessageBox::information(this , tr("Load Nodule") , tr("Load nodule file success."),QMessageBox::Ok);
        }
        else
        {
            QMessageBox::warning(this , tr("Load Nodule") , tr("Load nodule file failed."),QMessageBox::Ok);
        }
        //TODO check status
    }

    
}

void NoduleAnnotation::slot_sliding_bar_mpr00_i(int value)
{
    if(_model_crosshair->page_to(_mpr_scene_00 , value))
    {
        _model_crosshair->notify();
    }
}

void NoduleAnnotation::slot_sliding_bar_mpr01_i(int value)
{
    if(_model_crosshair->page_to(_mpr_scene_01 , value))
    {
        _model_crosshair->notify();
    }
}

void NoduleAnnotation::slot_sliding_bar_mpr10_i(int value)
{
    if(_model_crosshair->page_to(_mpr_scene_10 , value))
    {
        _model_crosshair->notify();
    }
}

void NoduleAnnotation::slot_voi_table_widget_cell_select_i(int row , int column)
{
    //std::cout << "CellSelect "<< row << " " << column<< std::endl; 
    VOISphere voi = _model_voi->get_voi_sphere(row);
    const Matrix4 mat_p2w = _mpr_scene_00->get_camera_calculator()->get_patient_to_world_matrix();
    _model_crosshair->locate(mat_p2w.transform(voi.center));
    _model_crosshair->notify();
    _select_vio_id = row;
}

void NoduleAnnotation::slot_voi_table_widget_item_changed_i(QTableWidgetItem *item)
{
    const int row = item->row();
    const int column = item->column();
    if (1 == column)
    {
        std::string sDiameter =  (item->text()).toLocal8Bit();
        StrNumConverter<double> con;
        _model_voi->modify_voi_sphere_diameter(row , con.to_num(sDiameter));
        _ob_scene_container->update();
    }
}

void NoduleAnnotation::slot_add_nodule_i()
{
    refresh_nodule_list_i();
}

void NoduleAnnotation::slot_delete_nodule_i()
{
    if (_select_vio_id >= 0 && _select_vio_id < _model_voi->get_voi_spheres().size())
    {
        _model_voi->remove_voi_sphere(_select_vio_id);
        _mpr_scene_00->set_dirty(true);
        _mpr_scene_01->set_dirty(true);
        _mpr_scene_10->set_dirty(true);
        refresh_nodule_list_i();
        _ob_scene_container->update();
        _select_vio_id = -1;
    }
}

void NoduleAnnotation::slot_voi_table_widget_nodule_type_changed_i(int id)
{
    QWidget* widget = ui.tableWidgetNoduleList->cellWidget(id , 2);

    QComboBox* pBox= dynamic_cast<QComboBox*>(widget);
    if (pBox)
    {
        std::string type = pBox->currentText().toStdString();
        std::cout << id <<'\t' << type << std::endl;

        _model_voi->modify_voi_sphere_name(id , type);
    }
}

void NoduleAnnotation::slot_preset_wl_changed_i(QString s)
{
    if (!_is_ready)
    {
        return;
    }

    std::string wl_preset = std::string(s.toLocal8Bit());
    float ww(1) , wl(0);
    if (wl_preset == std::string("CT_Lungs"))
    {
        ww = PRESET_CT_LUNGS_WW;
        wl   = PRESET_CT_LUNGS_WL;
    }
    else if (wl_preset == std::string("CT_Chest"))
    {
        ww = PRESET_CT_CHEST_WW;
        wl   = PRESET_CT_CHEST_WL;
    }
    else if (wl_preset == std::string("CT_Bone"))
    {
        ww = PRESET_CT_BONE_WW;
        wl   = PRESET_CT_BONE_WL;
    }
    else if (wl_preset == std::string("CT_Angio"))
    {
        ww = PRESET_CT_ANGIO_WW;
        wl   = PRESET_CT_ANGIO_WL;
    }
    else if (wl_preset == std::string("CT_Abdomen"))
    {
        ww = PRESET_CT_ABDOMEN_WW;
        wl   = PRESET_CT_ABDOMEN_WL;
    }
    else if (wl_preset == std::string("CT_Brain"))
    {
        ww = PRESET_CT_BRAIN_WW;
        wl   = PRESET_CT_BRAIN_WL;
    }
    else 
    {
        return;
    }

    _mpr_scene_00->set_global_window_level(ww , wl);
    _mpr_scene_01->set_global_window_level(ww , wl);
    _mpr_scene_10->set_global_window_level(ww , wl);
    _ob_scene_container->update();
}

void NoduleAnnotation::slot_scene_min_max_hint_i(const std::string& name)
{
    if (!_is_ready)
    {
        return;
    }

    SceneContainer* target_container = nullptr;
    QScrollBar* target_scroll_bar = nullptr;
    if (0 == _layout_tag)
    {
        save_layout2x2_parameter_i();

        if (name == _mpr_scene_00->get_name())
        {
            target_container = _mpr_00;
            target_scroll_bar = _mpr_00_scroll_bar;
        }
        else if (name == _mpr_scene_01->get_name())
        {
            target_container = _mpr_01;
            target_scroll_bar = _mpr_01_scroll_bar;
        }
        else if (name == _mpr_scene_10->get_name())
        {
            target_container = _mpr10;
            target_scroll_bar = _mpr_10_scroll_bar;
        }
        else
        {
            return;
        }

        _mpr_00->hide();
        _mpr_00_scroll_bar->hide();
        _mpr_01->hide();
        _mpr_01_scroll_bar->hide();
        _mpr10->hide();
        _mpr_10_scroll_bar->hide();
        _vr_11->hide();

        ui.gridLayout_6->removeWidget(_mpr_00);
        ui.gridLayout_6->removeWidget(_mpr_00_scroll_bar);
        ui.gridLayout_6->removeWidget(_mpr_01);
        ui.gridLayout_6->removeWidget(_mpr_01_scroll_bar);
        ui.gridLayout_6->removeWidget(_mpr10 );
        ui.gridLayout_6->removeWidget(_mpr_10_scroll_bar);
        ui.gridLayout_6->removeWidget(_vr_11);

        ui.gridLayout_6->addWidget(target_container , 0 ,0);
        ui.gridLayout_6->addWidget(target_scroll_bar , 0 ,1,1,1);

        target_container->show();
        target_scroll_bar->show();
        target_container->update_scene();

        _layout_tag = 1;

    }
    else
    {
        slot_change_layout2x2_i();
    }
}

void NoduleAnnotation::slot_focus_in_scene_i(QString s)
{
    if (!_is_ready)
    {
        return;
    }

    const std::string name(s.toLocal8Bit());

    if (name == _mpr_scene_00->get_name())
    {
        _model_crosshair->focus(_mpr_scene_00);
    }
    else if (name == _mpr_scene_01->get_name())
    {
        _model_crosshair->focus(_mpr_scene_01);
    }
    else if (name == _mpr_scene_10->get_name())
    {
        _model_crosshair->focus(_mpr_scene_10);
    }
    else
    {

    }
}

void NoduleAnnotation::slot_focus_out_scene_i(QString name)
{
    if (!_is_ready)
    {
        return;
    }
    _model_crosshair->focus(nullptr);

}

void NoduleAnnotation::slot_crosshair_visibility_i(int iFlag)
{
    if (!_is_ready)
    {
        return;
    }

    _model_crosshair->set_visibility(iFlag != 0);

    SceneContainer* containers[3] = {_mpr_00 , _mpr_01 , _mpr10};
    std::shared_ptr<MPRScene> scenes[3] = {_mpr_scene_00 , _mpr_scene_01 , _mpr_scene_10};
    if (0 == iFlag)//Hide
    {
        if (ui.pushButtonArrow->isChecked())
        {
            //Unregister mouse locate operation
            for (int i = 0 ;i<3 ; ++i)
            {
                IMouseOpPtrCollection ops = containers[i]->get_mouse_operation(Qt::LeftButton ,  Qt::NoModifier);
                IMouseOpPtrCollection ops_new;
                for (auto it = ops.begin() ; it != ops.end() ; ++it)
                {
                    if (!std::dynamic_pointer_cast<MouseOpLocate>(*it) )
                    {
                        ops_new.push_back(*it);
                    }
                }
                containers[i]->register_mouse_operation(ops_new , Qt::LeftButton , Qt::NoModifier);
            }
        }
    }
    else//Show
    {
        if (ui.pushButtonArrow->isChecked())
        {
            for (int i = 0 ;i<3 ; ++i)
            {
                IMouseOpPtrCollection ops = containers[i]->get_mouse_operation(Qt::LeftButton ,  Qt::NoModifier);
                IMouseOpPtrCollection ops_new;
                for (auto it = ops.begin() ; it != ops.end() ; ++it)
                {
                    if (!std::dynamic_pointer_cast<MouseOpLocate>(*it) )
                    {
                        ops_new.push_back(*it);
                    }
                }
                std::shared_ptr<MouseOpLocate> op_locate(new MouseOpLocate());
                op_locate->set_scene(scenes[i]);
                op_locate->set_crosshair_model(_model_crosshair);
                ops_new.push_back(op_locate);
                containers[i]->register_mouse_operation(ops_new , Qt::LeftButton , Qt::NoModifier);
            }
        }
    }
    _ob_scene_container->update();
}

void NoduleAnnotation::refresh_nodule_list_i()
{
    //reset nodule list
    ui.tableWidgetNoduleList->clear();
    ui.tableWidgetNoduleList->setRowCount(0);
    _select_vio_id = -1;

    const std::vector<VOISphere>& vois = _model_voi->get_voi_spheres();
    if (!vois.empty())
    {
        ui.tableWidgetNoduleList->setRowCount(vois.size());//Set row count , otherwise set item useless
        StrNumConverter<double> converter;
        const int iPrecision = 2;
        int iRow = 0;
        for (auto it = vois.begin() ; it != vois.end() ; ++it)
        {
            const VOISphere& voi = *it;
            std::string sPos = converter.to_string_decimal(voi.center.x , iPrecision) + "," +
                converter.to_string_decimal(voi.center.y , iPrecision) + "," +
                converter.to_string_decimal(voi.center.z , iPrecision);
            std::string sRadius = converter.to_string_decimal(voi.diameter , iPrecision);

            QTableWidgetItem* pPos= new QTableWidgetItem(sPos.c_str());
            pPos->setFlags(pPos->flags() & ~Qt::ItemIsEnabled);
            ui.tableWidgetNoduleList->setItem(iRow,0, pPos);
            ui.tableWidgetNoduleList->setItem(iRow,1, new QTableWidgetItem(sRadius.c_str()));

            QComboBox * pNoduleType = new QComboBox(ui.tableWidgetNoduleList);
            pNoduleType->clear();
            pNoduleType->insertItem(0 ,  NODULE_TYPE_GGN.c_str());
            pNoduleType->insertItem(1 , NODULE_TYPE_AAH.c_str());
            ui.tableWidgetNoduleList->setCellWidget(iRow,2, pNoduleType);

            connect(pNoduleType , SIGNAL(currentIndexChanged(int)) , _single_manager_nodule_type , SLOT(map()));
            _single_manager_nodule_type->setMapping(pNoduleType , iRow);

            ++iRow;
        }
    }
}

void NoduleAnnotation::save_layout2x2_parameter_i()
{
    _pre_2x2_width = _mpr_00->width();
    _pre_2x2_height = _mpr_00->height();
}





