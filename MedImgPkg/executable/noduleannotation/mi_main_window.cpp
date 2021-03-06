#include "mi_main_window.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include "io/mi_configure.h"
#include "util/mi_string_number_converter.h"
#include "util/mi_model_progress.h"

#include "arithmetic/mi_rsa_utils.h"
#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_run_length_operator.h"
#include "arithmetic/mi_arithmetic_utils.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"
#include "io/mi_meta_object_loader.h"
#include "io/mi_nodule_set.h"
#include "io/mi_nodule_set_parser.h"
#include "io/mi_mask_voi_converter.h"

#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_texture_cache.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_mpr_entry_exit_points.h"
#include "renderalgo/mi_ray_caster_canvas.h"
#include "renderalgo/mi_ray_caster.h"
#include "renderalgo/mi_camera_interactor.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_mask_label_store.h"

#include "qtpackage/mi_shared_widget.h"
#include "qtpackage/mi_scene_container.h"
#include "qtpackage/mi_graphic_item_corners_info.h"
#include "qtpackage/mi_graphic_item_direction_info.h"
#include "qtpackage/mi_graphic_item_voi.h"
#include "qtpackage/mi_graphic_item_cross_hair.h"
#include "qtpackage/mi_graphic_item_mpr_border.h"
#include "qtpackage/mi_mouse_op_zoom.h"
#include "qtpackage/mi_mouse_op_pan.h"
#include "qtpackage/mi_mouse_op_rotate.h"
#include "qtpackage/mi_mouse_op_mpr_page.h"
#include "qtpackage/mi_mouse_op_windowing.h"
#include "qtpackage/mi_mouse_op_probe.h"
#include "qtpackage/mi_mouse_op_annotate.h"
#include "qtpackage/mi_mouse_op_annotate_fine_tuning.h"
#include "qtpackage/mi_mouse_op_locate.h"
#include "qtpackage/mi_mouse_op_test.h"
#include "qtpackage/mi_model_voi.h"
#include "qtpackage/mi_model_cross_hair.h"
#include "qtpackage/mi_model_focus.h"
#include "qtpackage/mi_observer_scene_container.h"
#include "qtpackage/mi_observer_progress.h"
#include "qtpackage/mi_observer_voi_statistic.h"
#include "qtpackage/mi_observer_voi_segment.h"
#include "qtpackage/mi_graphic_item_voi.h"

#include "mi_observer_voi_table.h"
#include "mi_observer_mpr_scroll_bar.h"
#include "mi_mouse_op_min_max_hint.h"
#include "mi_my_rsa.h"
#include "mi_dicom_anonymization_dialog.h"
#include "mi_raw_data_import_dialog.h"
#include "mi_setting_dialog.h"
#include "mi_nodule_anno_config.h"
#include "mi_nodule_anno_logger.h"

#include <QEvent>
#include <QSizePolicy>
#include <QScrollBar>
#include <QFileDialog>
#include <QMessagebox>
#include <QSignalMapper>
#include <QProgressDialog>
#include <QKeyEvent>

using namespace medical_imaging;

//Nodule type
#define NODULE_TYPE_NUM 6
static const std::string S_NODULE_TYPES[NODULE_TYPE_NUM] = 
{
    "W",
    "V",
    "P",
    "J",
    "G",
    "N"
};
static const std::string S_NODULE_TYPE_DESCRIPTION[NODULE_TYPE_NUM] = 
{
    "Well-Circumscribed",
    "Vascualarized ",
    "Pleural-Tail",
    "Juxta-Pleural",
    "GGO",
    "Non-nodule"
};

const std::string CONFIG_FILE = "./config/app_config";

NoduleAnnotation::NoduleAnnotation(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags),
    _layout_tag(0),
    _is_ready(false),
    _object_nodule(nullptr),
    _single_manager_nodule_type(nullptr),
    _select_voi_id(-1)
{
    _ui.setupUi(this);

    _ui.pushButtonFineTune->setVisible(false);
    _ui.comboBoxTuneType->setVisible(false);
    _ui.spinBoxTuneRadius->setVisible(false);

    _ui.tableWidgetNoduleList->setSelectionBehavior(QAbstractItemView::SelectRows);
    _ui.tableWidgetNoduleList->setSelectionMode(QAbstractItemView::SingleSelection);
    _ui.tableWidgetNoduleList->installEventFilter(this);

    _mpr_00 = new SceneContainer(SharedWidget::instance());
    _mpr_01 = new SceneContainer(SharedWidget::instance());
    _mpr_10 = new SceneContainer(SharedWidget::instance());
    _vr_11 = new SceneContainer(SharedWidget::instance());

    _mpr_10->setMinimumSize(100,100);
    _mpr_01->setMinimumSize(100,100);
    _mpr_00->setMinimumSize(100,100);
    _vr_11->setMinimumSize(100,100);

    //_mpr_00->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);//自适应窗口
    //_mpr_01->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);
    //_mpr10->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);
    //_vr_11->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);


    _mpr_00_scroll_bar = new QScrollBar(_ui.centralWidget);
    _mpr_00_scroll_bar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR00"));
    _mpr_00_scroll_bar->setOrientation(Qt::Vertical);

    _mpr_01_scroll_bar = new QScrollBar(_ui.centralWidget);
    _mpr_01_scroll_bar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR01"));
    _mpr_01_scroll_bar->setOrientation(Qt::Vertical);

    _mpr_10_scroll_bar = new QScrollBar(_ui.centralWidget);
    _mpr_10_scroll_bar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR10"));
    _mpr_10_scroll_bar->setOrientation(Qt::Vertical);


    _ui.gridLayout_6->addWidget(_mpr_00 , 0 ,0);
    _ui.gridLayout_6->addWidget(_mpr_00_scroll_bar , 0 ,1,1,1);
    _ui.gridLayout_6->addWidget(_mpr_01 , 0 ,2);
    _ui.gridLayout_6->addWidget(_mpr_01_scroll_bar , 0 ,3,1,1);
    _ui.gridLayout_6->addWidget(_mpr_10 , 1 ,0);
    _ui.gridLayout_6->addWidget(_mpr_10_scroll_bar , 1 ,1,1,1);
    _ui.gridLayout_6->addWidget(_vr_11 , 1 ,2);

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
    NoduleAnnoConfig::instance()->bind_config_file(CONFIG_FILE);
    NoduleAnnoConfig::instance()->initialize();
    GLUtils::set_check_gl_flag(false);
}

void NoduleAnnotation::create_scene_i()
{
    int interval = NoduleAnnoConfig::instance()->get_double_click_interval();
    _mpr_00->set_double_click_interval(interval);
    _mpr_scene_00.reset(new MPRScene(_mpr_00->width() , _mpr_00->height()));
    
    _mpr_01->set_double_click_interval(interval);
    _mpr_scene_01.reset(new MPRScene(_mpr_01->width() , _mpr_01->height()));

    _mpr_10->set_double_click_interval(interval);
    _mpr_scene_10.reset(new MPRScene(_mpr_10->width() , _mpr_10->height()));

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr_10);

    //Set scenes
    _ob_voi_segment->set_scenes(mpr_scenes);

    for (int i = 0 ; i < 3 ; ++i)
    {
        //1 Set Scene
        mpr_containers[i]->set_scene(mpr_scenes[i]);

        //2 Set scene parameter
        mpr_scenes[i]->set_mask_label_level(L_64);
        mpr_scenes[i]->set_volume_infos(_volume_infos);
        mpr_scenes[i]->set_sample_rate(1.0);
        float ww(0.0f), wl(0.0f);
        NoduleAnnoConfig::instance()->get_preset_wl(CT_LUNGS, ww, wl);
        mpr_scenes[i]->set_global_window_level(ww,wl);
        mpr_scenes[i]->set_composite_mode(COMPOSITE_AVERAGE);
        mpr_scenes[i]->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
        mpr_scenes[i]->set_mask_mode(MASK_NONE);
        mpr_scenes[i]->set_interpolation_mode(LINEAR);

        //3 Add painter list
        std::shared_ptr<GraphicItemCornersInfo> graphic_item_corner_info(new GraphicItemCornersInfo());
        graphic_item_corner_info->set_scene(mpr_scenes[i]);
        mpr_containers[i]->add_item(graphic_item_corner_info);

        std::shared_ptr<GraphicItemDirectionInfo> graphic_item_direction_info(new GraphicItemDirectionInfo());
        graphic_item_direction_info->set_scene(mpr_scenes[i]);
        mpr_containers[i]->add_item(graphic_item_direction_info);

        std::shared_ptr<GraphicItemVOI> graphic_item_voi(new GraphicItemVOI());
        graphic_item_voi->set_scene(mpr_scenes[i]);
        graphic_item_voi->set_voi_model(_model_voi);
        mpr_containers[i]->add_item(graphic_item_voi);
        this->_voi_collections.push_back(graphic_item_voi);

        std::shared_ptr<GraphicItemCrosshair> graphic_item_crosshair(new GraphicItemCrosshair());
        graphic_item_crosshair->set_scene(mpr_scenes[i]);
        graphic_item_crosshair->set_crosshair_model(_model_crosshair);
        mpr_containers[i]->add_item(graphic_item_crosshair);

        std::shared_ptr<GraphicItemMPRBorder> graphic_item_mpr_border(new GraphicItemMPRBorder());
        graphic_item_mpr_border->set_scene(mpr_scenes[i]);
        graphic_item_mpr_border->set_crosshair_model(_model_crosshair);
        graphic_item_mpr_border->set_focus_model(_model_focus);
        mpr_containers[i]->add_item(graphic_item_mpr_border);

        //4 Add operation 
        std::shared_ptr<MouseOpMinMaxHint> op_min_max_hint(new MouseOpMinMaxHint());
        op_min_max_hint->set_scene(mpr_scenes[i]);
        op_min_max_hint->set_min_max_hint_object(_object_min_max_hint);
        mpr_containers[i]->register_mouse_double_click_operation(op_min_max_hint);

        //std::shared_ptr<MouseOpLocate> op_mpr_locate(new MouseOpLocate());
        //op_mpr_locate->set_scene(mpr_scenes[i]);
        //op_mpr_locate->set_crosshair_model(_model_crosshair);
        //mpr_containers[i]->register_mouse_operation(op_mpr_locate, Qt::LeftButton , Qt::NoModifier);

        std::shared_ptr<MouseOpZoom> op_zoom(new MouseOpZoom());
        op_zoom->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_zoom , Qt::RightButton , Qt::NoModifier);

        std::shared_ptr<MouseOpWindowing> op_windowing(new MouseOpWindowing());
        op_windowing->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_windowing , Qt::MiddleButton , Qt::NoModifier);

        //////////////////////////////////////////////////////////////////////////
        //Debug middle mouse to test
        /*std::shared_ptr<MouseOpTest> op_test(new MouseOpTest());
        op_test->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_test , Qt::MiddleButton , Qt::NoModifier);*/
        //////////////////////////////////////////////////////////////////////////

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
    // Just connect focus in scene
    QSignalMapper* focus_in_singal_mapper = new QSignalMapper();

    connect(_mpr_00 , SIGNAL(focus_in_scene()) , focus_in_singal_mapper , SLOT(map()));
    focus_in_singal_mapper->setMapping(_mpr_00 , QString(_mpr_00->get_name().c_str()));
    connect(_mpr_01 , SIGNAL(focus_in_scene()) , focus_in_singal_mapper , SLOT(map()));
    focus_in_singal_mapper->setMapping(_mpr_01 , QString(_mpr_01->get_name().c_str()));
    connect(_mpr_10 , SIGNAL(focus_in_scene()) , focus_in_singal_mapper , SLOT(map()));
    focus_in_singal_mapper->setMapping(_mpr_10 , QString(_mpr_10->get_name().c_str()));

    connect(focus_in_singal_mapper , SIGNAL(mapped(QString)) , this , SLOT(slot_focus_in_scene_i(QString)));

    /*QSignalMapper* focus_out_singal_mapper = new QSignalMapper();

    connect(_mpr_00 , SIGNAL(focus_out_scene()) , focus_out_singal_mapper , SLOT(map()));
    focus_out_singal_mapper->setMapping(_mpr_00 , QString(_mpr_00->get_name().c_str()));
    connect(_mpr_01 , SIGNAL(focus_out_scene()) , focus_out_singal_mapper , SLOT(map()));
    focus_out_singal_mapper->setMapping(_mpr_01 , QString(_mpr_01->get_name().c_str()));
    connect(_mpr10 , SIGNAL(focus_out_scene()) , focus_out_singal_mapper , SLOT(map()));
    focus_out_singal_mapper->setMapping(_mpr10 , QString(_mpr10->get_name().c_str()));

    connect(focus_out_singal_mapper , SIGNAL(mapped(QString)) , this , SLOT(slot_focus_out_scene_i(QString)));*/
    //////////////////////////////////////////////////////////////////////////

}

void NoduleAnnotation::connect_signal_slot_i()
{
    //Layout
    //connect(ui.action1x1 , SIGNAL(triggered()) , this , SLOT(SlotChangeLayout1x1_i()));
    connect(_ui.action2x2 , SIGNAL(triggered()) , this , SLOT(slot_change_layout2x2_i()));

    // Layout three 2D views
    connect(_ui.action2D_Sagittal , SIGNAL(triggered()) , this , SLOT(slot_change_layout1x1_i()));
    connect(_ui.action2D_Coronal , SIGNAL(triggered()) , this , SLOT(slot_change_layout1x1_i()));
    connect(_ui.action2D_Tranverse , SIGNAL(triggered()) , this , SLOT(slot_change_layout1x1_i()));

    //File
    connect(_ui.actionOpen_DICOM_Folder , SIGNAL(triggered()) , this , SLOT(slot_open_dicom_folder_i()));
    connect(_ui.actionOpen_Meta_Image , SIGNAL(triggered()) , this , SLOT(slot_open_meta_image_i()));
    connect(_ui.actionOpen_Raw , SIGNAL(triggered()) , this , SLOT(slot_open_raw_i()));
    connect(_ui.actionSave_Nodule , SIGNAL(triggered()) , this , SLOT(slot_save_nodule_file_i()));
    connect(_ui.actionLoad_Nodule , SIGNAL(triggered()) , this , SLOT(slot_open_nodule_file_i()));
    connect(_ui.actionAnonymization_DICOM , SIGNAL(triggered()) , this , SLOT(slot_dicom_anonymization_i()));
    connect(_ui.actionQuit , SIGNAL(triggered()) , this , SLOT(slot_quit_i()));

    connect(_ui.actionSave_Label, SIGNAL(triggered()) , this , SLOT(slot_save_label_file()));
    connect(_ui.actionLoad_Label, SIGNAL(triggered()) , this , SLOT(slot_load_label_file()));

    //MPR scroll bar
    connect(_mpr_00_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr00_i(int)));
    connect(_mpr_01_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr01_i(int)));
    connect(_mpr_10_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr10_i(int)));

    //Common tools
    connect(_ui.pushButtonArrow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_arrow_i()));
    connect(_ui.pushButtonAnnotate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_annotate_i()));
    connect(_ui.pushButtonLocate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_locate_i()));
    connect(_ui.pushButtonZoom , SIGNAL(pressed()) , this , SLOT(slot_press_btn_zoom_i()));
    connect(_ui.pushButtonPan , SIGNAL(pressed()) , this , SLOT(slot_press_btn_pan_i()));
    connect(_ui.pushButtonWindowing , SIGNAL(pressed()) , this , SLOT(slot_press_btn_windowing_i()));
    connect(_ui.pushButtonFitWindow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_fit_window_i()));

    connect(_ui.pushButtonFineTune , SIGNAL(pressed()) , this , SLOT(slot_press_btn_fine_tune_i()));
    connect(_ui.spinBoxTuneRadius , SIGNAL(valueChanged(int)) , this , SLOT(slot_spn_box_tune_radius(int)));

    //VOI list
    connect(_ui.tableWidgetNoduleList , SIGNAL(cellPressed(int,int)) , this , SLOT(slot_voi_table_widget_cell_select_i(int ,int)));
    connect(_ui.tableWidgetNoduleList , SIGNAL(itemChanged(QTableWidgetItem *)) , this , SLOT(slot_voi_table_widget_item_changed_i(QTableWidgetItem *)));
    connect(_object_nodule , SIGNAL(nodule_added()) , this , SLOT(slot_add_nodule_i()));
    connect(_ui.pushButtonDeleteNodule , SIGNAL(pressed()) , this , SLOT(slot_delete_nodule_i()));

    //Preset WL
    connect(_ui.comboBoxPresetWL , SIGNAL(currentIndexChanged(QString)) , this , SLOT(slot_preset_wl_changed_i(QString)));

    //Scene Min Max hint
    connect(_object_min_max_hint , SIGNAL(triggered(const std::string&)) , this , SLOT(slot_scene_min_max_hint_i(const std::string&)));

    //Crosshair visibility
    connect(_ui.checkBoxCrossHair , SIGNAL(stateChanged(int)) , this , SLOT(slot_crosshair_visibility_i(int)));

    //Nodule overlay visibility
    connect(_ui.checkBoxNoduleOverlay , SIGNAL(stateChanged(int)) , this , SLOT(slot_nodule_overlay_visibility_i(int)));

    //Setting
    connect(_ui.actionSetting, SIGNAL(triggered()), this, SLOT(slot_open_setting_dlg()) );
}

void NoduleAnnotation::create_model_observer_i()
{
    //VOI
    _model_voi.reset(new VOIModel());

    _ob_voi_table.reset(new VOITableObserver());
    _ob_voi_table->set_nodule_object(_object_nodule);

    _ob_scene_container.reset(new SceneContainerObserver());
    _ob_scene_container->add_scene_container(_mpr_00);
    _ob_scene_container->add_scene_container(_mpr_01);
    _ob_scene_container->add_scene_container(_mpr_10);

    _ob_voi_statistic.reset(new VOIStatisticObserver());
    _ob_voi_statistic->set_model(_model_voi);
    _ob_voi_statistic->set_volume_infos(_volume_infos);

    _ob_voi_segment.reset(new VOISegmentObserver());
    _ob_voi_segment->set_model(_model_voi);
    _ob_voi_segment->set_volume_infos(_volume_infos);

    _model_voi->add_observer(_ob_voi_table);
    _model_voi->add_observer(_ob_voi_segment);//Segment first
    _model_voi->add_observer(_ob_voi_statistic);// Then statistic

    //m_painter_voiModel->add_observer(m_pSceneContainerOb);//Scene refresh called by change item

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

    //Focus model
    _model_focus.reset(new FocusModel());
    _model_focus->set_focus_scene_container(_mpr_00);//Default focus MPR00
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
    _mpr_10->hide();
    _mpr_10_scroll_bar->hide();
    _vr_11->hide();

    _ui.gridLayout_6->removeWidget(_mpr_00);
    _ui.gridLayout_6->removeWidget(_mpr_00_scroll_bar);
    _ui.gridLayout_6->removeWidget(_mpr_01);
    _ui.gridLayout_6->removeWidget(_mpr_01_scroll_bar);
    _ui.gridLayout_6->removeWidget(_mpr_10 );
    _ui.gridLayout_6->removeWidget(_mpr_10_scroll_bar);
    _ui.gridLayout_6->removeWidget(_vr_11);

    _ui.gridLayout_6->addWidget(_mpr_00 , 0 ,0);
    _ui.gridLayout_6->addWidget(_mpr_00_scroll_bar , 0 ,1,1,1);
    _ui.gridLayout_6->addWidget(_mpr_01 , 0 ,2);
    _ui.gridLayout_6->addWidget(_mpr_01_scroll_bar , 0 ,3,1,1);
    _ui.gridLayout_6->addWidget(_mpr_10 , 1 ,0);
    _ui.gridLayout_6->addWidget(_mpr_10_scroll_bar , 1 ,1,1,1);
    _ui.gridLayout_6->addWidget(_vr_11 , 1 ,2);

    //Set min size to fix size bug
    _mpr_00->setMinimumSize(_pre_2x2_width , _pre_2x2_height);
    _mpr_01->setMinimumSize(_pre_2x2_width , _pre_2x2_height);
    _mpr_10->setMinimumSize(_pre_2x2_width , _pre_2x2_height);
    _vr_11->setMinimumSize(_pre_2x2_width ,_pre_2x2_height);


    _mpr_00->show();
    _mpr_00_scroll_bar->show();
    _mpr_01->show();
    _mpr_01_scroll_bar->show();
    _mpr_10->show();
    _mpr_10_scroll_bar->show();
    _vr_11->show();

    //Recover min size to expanding
    _mpr_10->setMinimumSize(100,100);
    _mpr_01->setMinimumSize(100,100);
    _mpr_00->setMinimumSize(100,100);
    _vr_11->setMinimumSize(100,100);

    _layout_tag = 0;
}

void NoduleAnnotation::slot_change_layout1x1_i()
{
    if (!_is_ready)
    {
        return;
    }

    if (this->sender() == this->_ui.action2D_Sagittal)
    {
        change_layout1x1_i(SAGITTAL);
    }
    else if (this->sender() == this->_ui.action2D_Coronal)
    {
        change_layout1x1_i(CORONAL);
    }
    else /*if (this->sender() == this->_ui.action2D_Transverse)*/
    {
        change_layout1x1_i(TRANSVERSE);
    }
}

void NoduleAnnotation::slot_open_dicom_folder_i()
{
    QStringList file_name_list = QFileDialog::getOpenFileNames(
        this ,tr("Loading DICOM Dialog"),NoduleAnnoConfig::instance()->get_last_open_direction().c_str(),
        tr("Dicom image(*dcm);;Other(*)"));

    if (!file_name_list.empty())
    {
        update_last_open_direction_i(std::string(file_name_list[0].toLocal8Bit()));

        QApplication::setOverrideCursor(Qt::WaitCursor);

        //Init progress dialog
        std::shared_ptr<ProgressObserver> progress_ob(new ProgressObserver());
        QProgressDialog progress_dialog(tr("Loading DICOM series ......") ,0 , 0 , 100 );

        _model_progress->clear_observer();
        _model_progress->add_observer(progress_ob);
        progress_ob->set_progress_model(_model_progress);
        progress_ob->set_progress_dialog(&progress_dialog);


        progress_dialog.setWindowTitle(tr("please wait."));
        progress_dialog.setFixedWidth(300);
        progress_dialog.setWindowModality(Qt::WindowModal);
        progress_dialog.show();

        std::vector<std::string> file_names_std(file_name_list.size());
        int idx = 0;
        for (auto it = file_name_list.begin() ; it != file_name_list.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            file_names_std[idx++] = s;
        }

        std::shared_ptr<ImageDataHeader> data_header;
        std::shared_ptr<ImageData> img_data;
        DICOMLoader loader;
        loader.set_progress_model(_model_progress);
        IOStatus status = loader.load_series(file_names_std, img_data , data_header);
        if (status != IO_SUCCESS)
        {
            QApplication::restoreOverrideCursor();
            QMessageBox::warning(this , tr("Load DICOM folder") , tr("load DICOM folder failed!"));
            _model_progress->clear_observer();
            return;
        }

        //Set DICOM series files
        _dicom_series_files.clear();
        _dicom_series_files = file_names_std;

        //Load data
        load_data_i(img_data , data_header);

        QApplication::restoreOverrideCursor();
    }
    else
    {
        return;
    }
}

void NoduleAnnotation::release_i()
{
    _volume_infos.reset();
    //release scene
    /* if (_mpr_00)
    {
    delete _mpr_00;
    _mpr_00 = nullptr;
    }

    if (_mpr_01)
    {
    delete _mpr_01;
    _mpr_01 = nullptr;
    }

    if (_mpr_10)
    {
    delete _mpr_10;
    _mpr_10 = nullptr;
    }*/

    _mpr_scene_00.reset();
    _mpr_scene_01.reset();
    _mpr_scene_10.reset();

    if (_vr_11)
    {
        delete _vr_11;
        _vr_11 = nullptr;
    }

    _model_voi.reset();
    _model_crosshair.reset();
    _model_progress.reset();
    _model_focus.reset();

    _ob_voi_table.reset();
    _ob_voi_statistic.reset();
    _ob_voi_segment.reset();
    _ob_scene_container.reset();
    _ob_mpr_scroll_bar.reset();

    _voi_collections.clear();
    _dicom_series_files.clear();

    //GLTextureCache::instance()->process_cache();
    //GLResourceManagerContainer::instance()->update_all();
}

void NoduleAnnotation::load_data_i(std::shared_ptr<ImageData> img_data ,std::shared_ptr<ImageDataHeader> data_header)
{
    //////////////////////////////////////////////////////////////////////////
    //Initialize
    if (_volume_infos)//Delete last one
    {
        _volume_infos->finialize();
    }
    _volume_infos.reset(new VolumeInfos());
    _volume_infos->set_data_header(data_header);
    //SharedWidget::instance()->makeCurrent();
    _volume_infos->set_volume(img_data);//load volume texture if has graphic card

    //Create empty mask
    std::shared_ptr<ImageData> mask_data(new ImageData());
    img_data->shallow_copy(mask_data.get());
    mask_data->_channel_num = 1;
    mask_data->_data_type = medical_imaging::UCHAR;
    mask_data->mem_allocate();
    _volume_infos->set_mask(mask_data);

    create_model_observer_i();

    _mpr_00->clear();
    _mpr_01->clear();
    _mpr_10->clear();

    create_scene_i();

    save_layout2x2_parameter_i();

    _mpr_00->update();
    _mpr_01->update();
    _mpr_10->update();

    _model_progress->clear_observer();

    //reset nodule list
    refresh_nodule_list_i();
    _select_voi_id = -1;

    _is_ready = true;
}

void NoduleAnnotation::slot_open_setting_dlg()
{
    SettingDlg *dlg = new SettingDlg();
    dlg->setWindowModality(Qt::WindowModal);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    float ww = 0.0f, wl = 0.0f;
    for (int i=0; i<CT_Preset_ALL; ++i)
    {
        NoduleAnnoConfig::instance()->get_preset_wl(PreSetWLType(i), ww, wl);
        dlg->set_preset_wl(PreSetWLType(i), ww, wl);
    }
    connect(dlg, SIGNAL(save_setting(std::vector<float>)), this, SLOT(slot_save_setting(std::vector<float>)));
    dlg->show();
}

void NoduleAnnotation::slot_save_setting(std::vector<float> ww_wl)
{
    for (int i=0; i<CT_Preset_ALL; ++i)
    {
        float ww = ww_wl.at(2*i+0);
        float wl = ww_wl.at(2*i+1);
        NoduleAnnoConfig::instance()->set_preset_wl(PreSetWLType(i), ww, wl);
    }
    NoduleAnnoConfig::instance()->finalize();
    slot_preset_wl_changed_i(this->_ui.comboBoxPresetWL->currentText());
}

bool NoduleAnnotation::eventFilter(QObject *object, QEvent *event)
{
    if (object == this->_ui.tableWidgetNoduleList && event->type() == QEvent::KeyPress) {
        QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
        if (keyEvent->key() == Qt::Key_Delete) {
            // Special tab handling
            this->slot_delete_nodule_i();
            return true;
        } else
            return false;
    }
    return QMainWindow::eventFilter(object, event);
}

void NoduleAnnotation::slot_dicom_anonymization_i()
{
    if (!_dicom_series_files.empty())
    {
        DICOMAnonymizationDlg *dlg = new DICOMAnonymizationDlg();
        dlg->setWindowModality(Qt::WindowModal);
        dlg->set_dicom_series_files(_dicom_series_files);
        dlg->set_progress_model(_model_progress);
        dlg->show();
    }
}

void NoduleAnnotation::slot_open_meta_image_i()
{

    QString file_name = QFileDialog::getOpenFileName(
        this ,tr("Loading meta data"),NoduleAnnoConfig::instance()->get_last_open_direction().c_str(),
        tr("Dicom image(*mhd)"));

    if (file_name.isEmpty())
    {
        return;
    }

    update_last_open_direction_i(std::string(file_name.toLocal8Bit()));

    QApplication::setOverrideCursor(Qt::WaitCursor);

    MetaObjectLoader loader;
    std::shared_ptr<MetaObjectTag> meta_tag;
    std::shared_ptr<ImageData> img_data;
    std::shared_ptr<ImageDataHeader> data_header;

    IOStatus status = loader.load(std::string(file_name.toLocal8Bit()) , img_data , meta_tag,data_header);
    if (status != IO_SUCCESS)
    {
        QApplication::restoreOverrideCursor();
        QMessageBox::warning(this , tr("Load meta data") , tr("load meta data failed!"));
        return;
    }

    //Clear DICOM series files
    _dicom_series_files.clear();

    load_data_i(img_data , data_header);

    QApplication::restoreOverrideCursor();
}

void NoduleAnnotation::slot_open_raw_i()
{
    RawDataImportDlg *dlg = new RawDataImportDlg();
    dlg->setWindowModality(Qt::WindowModal);

    connect(
        dlg , SIGNAL(raw_data_imported(std::shared_ptr<medical_imaging::ImageData> ,std::shared_ptr<medical_imaging::ImageDataHeader> )) , 
        this , SLOT(load_data_i(std::shared_ptr<medical_imaging::ImageData> ,std::shared_ptr<medical_imaging::ImageDataHeader> )) );

    dlg->show();
}

void NoduleAnnotation::slot_press_btn_annotate_i()
{
    if (!_is_ready)
    {
        return;
    }
    this->_current_operation = "Annotate";
    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr_10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpAnnotate> op_annotate(new MouseOpAnnotate());
        op_annotate->set_scene(mpr_scenes[i]);
        op_annotate->set_voi_model(_model_voi);//Set Model to annotate tools       
        mpr_containers[i]->register_mouse_operation(op_annotate , Qt::LeftButton , Qt::NoModifier);

        this->_voi_collections[i]->enable_interaction();
        this->_voi_collections[i]->set_item_to_be_tuned(-1);
        
        mpr_containers[i]->setMouseTracking(false);
    }

    this->_model_voi->set_voi_to_tune(-1);
}

void NoduleAnnotation::slot_press_btn_fine_tune_i()
{
    if (!_is_ready || this->_model_voi->get_vois().size() < 1)
    {
        return;
    }
    this->_current_operation = "Finetune";
    std::vector<MPRScenePtr> mpr_slices;
    mpr_slices.push_back(this->_mpr_scene_00);
    mpr_slices.push_back(this->_mpr_scene_01);
    mpr_slices.push_back(this->_mpr_scene_10);

    std::vector<SceneContainer*> render_windows;
    render_windows.push_back(_mpr_00);
    render_windows.push_back(_mpr_01);
    render_windows.push_back(_mpr_10);

    // can select the table entry but fail to highlight
    int voi_to_be_tuned = this->_select_voi_id == -1 ? this->_ui.tableWidgetNoduleList->rowCount()-1 : this->_select_voi_id;
    this->_ui.tableWidgetNoduleList->selectRow(voi_to_be_tuned);
    
    for (int i = 0 ; i < 3 ; ++i)
    {
        // take over left-button operation
        std::shared_ptr<MouseOpAnnotateFineTuning> op_annotate_fine_tuning(new MouseOpAnnotateFineTuning());
        op_annotate_fine_tuning->set_scene(mpr_slices[i]);
        op_annotate_fine_tuning->set_voi_model(this->_model_voi);//Set Model to annotate tools
        
        IMouseOpPtrCollection left_btn_ops(1);
        left_btn_ops[0] = op_annotate_fine_tuning;

        render_windows[i]->register_mouse_operation(left_btn_ops , Qt::LeftButton , Qt::NoModifier);
        render_windows[i]->register_mouse_operation(left_btn_ops , Qt::NoButton , Qt::NoModifier);

        // freeze all the vois
        this->_voi_collections[i]->disable_interaction();

        // highlight the circle
        this->_voi_collections[i]->set_item_to_be_tuned(voi_to_be_tuned);

        // make sure the circle updated
        // render_windows[i]->update_scene(); // not working :(
        
        render_windows[i]->set_mouse_hovering(true);
    }

    this->_model_voi->set_voi_to_tune(voi_to_be_tuned);
}

void NoduleAnnotation::slot_spn_box_tune_radius(int new_value)
{
    if (!_is_ready || !_model_voi)
    {
        return;
    }

    this->_model_voi->set_tune_radius(new_value);
}

void NoduleAnnotation::shift_tune_object()
{
    int voi_to_be_tuned = this->_select_voi_id == -1 ? this->_ui.tableWidgetNoduleList->rowCount()-1 : this->_select_voi_id;
    this->_ui.tableWidgetNoduleList->selectRow(voi_to_be_tuned);

    for (int view_idx=0; view_idx<3; ++view_idx)
    {
        this->_voi_collections[view_idx]->set_item_to_be_tuned(voi_to_be_tuned);
    }
    this->_model_voi->set_voi_to_tune(voi_to_be_tuned);
}

void NoduleAnnotation::slot_press_btn_arrow_i()
{
    if (!_is_ready)
    {
        return;
    }
    if (this->_current_operation.compare("Arrow") == 0)
    {
        return;
    }

    this->_current_operation = "Arrow";
    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr_10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        //std::shared_ptr<MouseOpLocate> op_mpr_locate(new MouseOpLocate());
        //op_mpr_locate->set_scene(mpr_scenes[i]);
        //op_mpr_locate->set_crosshair_model(_model_crosshair);
        mpr_containers[i]->register_mouse_operation(nullptr , Qt::LeftButton , Qt::NoModifier);
        mpr_containers[i]->set_mouse_hovering(false);
    }
}

void NoduleAnnotation::slot_press_btn_locate_i()
{
    if (!_is_ready)
    {
        return;
    }

    if (this->_current_operation.compare("Locate")==0)
    {
        return;
    }

    this->_current_operation = "Locate";
    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr_10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpLocate> op_locate(new MouseOpLocate());
        op_locate->set_scene(mpr_scenes[i]);
        op_locate->set_crosshair_model(_model_crosshair);
        mpr_containers[i]->register_mouse_operation(op_locate , Qt::LeftButton , Qt::NoModifier);
        mpr_containers[i]->set_mouse_hovering(false);
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
    mpr_containers.push_back(_mpr_10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpZoom> op_zoom(new MouseOpZoom());
        op_zoom->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_zoom , Qt::LeftButton , Qt::NoModifier);mpr_containers[i]->set_mouse_hovering(false);
    }
}

void NoduleAnnotation::slot_press_btn_pan_i()
{
    if (!_is_ready)
    {
        return;
    }

    if (this->_current_operation.compare("Zoom") == 0)
    {
        return;
    }
    this->_current_operation = "Zoom";

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr_10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpPan> op_pan(new MouseOpPan());
        op_pan->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_pan , Qt::LeftButton , Qt::NoModifier);mpr_containers[i]->set_mouse_hovering(false);
    }
}

void NoduleAnnotation::slot_press_btn_windowing_i()
{
    if (!_is_ready)
    {
        return;
    }

    if (this->_current_operation.compare("Windowing") == 0)
    {
        return;
    }
    this->_current_operation = "Windowing";

    std::vector<MPRScenePtr> mpr_scenes;
    mpr_scenes.push_back(_mpr_scene_00);
    mpr_scenes.push_back(_mpr_scene_01);
    mpr_scenes.push_back(_mpr_scene_10);

    std::vector<SceneContainer*> mpr_containers;
    mpr_containers.push_back(_mpr_00);
    mpr_containers.push_back(_mpr_01);
    mpr_containers.push_back(_mpr_10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpWindowing> op_windowing(new MouseOpWindowing());
        op_windowing->set_scene(mpr_scenes[i]);
        mpr_containers[i]->register_mouse_operation(op_windowing , Qt::LeftButton , Qt::NoModifier);mpr_containers[i]->set_mouse_hovering(false);
    }
}

void NoduleAnnotation::slot_press_btn_fit_window_i()
{
    //TODO
    if (!_is_ready)
    {
        return;
    }

    if (_model_focus->get_focus_scene_container() == _mpr_00)
    {
        _mpr_scene_00->place_mpr(SAGITTAL);
        _model_crosshair->set_changed();
        _model_voi->set_changed();
    }
    else if (_model_focus->get_focus_scene_container() == _mpr_01)
    {
        _mpr_scene_01->place_mpr(CORONAL);
        _model_crosshair->set_changed();
        _model_voi->set_changed();
    }
    else if (_model_focus->get_focus_scene_container() == _mpr_10)
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

    if (_model_voi->get_vois().empty())
    {
        if(QMessageBox::No == QMessageBox::warning(
            this , tr("Save Nodule") , tr("Nodule count is zero. If you still want to save to file?"),QMessageBox::Yes |QMessageBox::No))
        {
            return;
        }
    }

    QString file_name = NoduleAnnoConfig::instance()->get_nodule_file_rsa() ?
        QFileDialog::getSaveFileName(this, tr("Save Nodule") , (NoduleAnnoConfig::instance()->get_last_open_direction() + "/" +_volume_infos->get_data_header()->series_uid).c_str(), tr("NoduleSet(*.nraw)")) :
        QFileDialog::getSaveFileName(this, tr("Save Nodule") , (NoduleAnnoConfig::instance()->get_last_open_direction() + "/" +_volume_infos->get_data_header()->series_uid).c_str(), tr("NoduleSet(*.nraw);;NoduleSet(*.csv)"));

    if (!file_name.isEmpty())
    {
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        const std::vector<VOISphere>& vois = _model_voi->get_vois();
        nodule_set->set_nodule(vois);

        NoduleSetParser parser;
        parser.set_series_id(_volume_infos->get_data_header()->series_uid);
        std::string file_name_std(file_name.toLocal8Bit());

        IOStatus status;
        const bool is_csv = file_name_std.substr(file_name_std.size() - 3 , 3) == std::string("csv");
        if (!is_csv)
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

namespace 
{
    int _check_nodule_valid(std::shared_ptr<ImageData> volume_data,
        std::shared_ptr<CameraCalculator> camera_cal,
        const VOISphere& voi)
    {
        const Matrix4& mat_p2w = camera_cal->get_patient_to_world_matrix();
        const Matrix4& mat_w2v = camera_cal->get_world_to_volume_matrix();
        Matrix4 mat_p2v = mat_w2v*mat_p2w;

        PatientAxisInfo head_info = camera_cal->get_head_patient_axis_info();
        PatientAxisInfo posterior_info = camera_cal->get_posterior_patient_axis_info();
        PatientAxisInfo left_info = camera_cal->get_left_patient_axis_info();
        double basic_abc[3];
        basic_abc[head_info.volume_coord/2] = volume_data->_spacing[head_info.volume_coord/2];
        basic_abc[posterior_info.volume_coord/2] = volume_data->_spacing[posterior_info.volume_coord/2];
        basic_abc[left_info.volume_coord/2] = volume_data->_spacing[left_info.volume_coord/2];

        Ellipsoid ellipsoid;
        ellipsoid._center = mat_p2v.transform(voi.center);
        double voi_abc[3] = {0,0,0};
        voi_abc[head_info.volume_coord/2] = voi.diameter*0.5/basic_abc[head_info.volume_coord/2] ;
        voi_abc[left_info.volume_coord/2] = voi.diameter*0.5/basic_abc[left_info.volume_coord/2] ;
        voi_abc[posterior_info.volume_coord/2] = voi.diameter*0.5/basic_abc[posterior_info.volume_coord/2] ;
        ellipsoid._a = voi_abc[0];
        ellipsoid._b = voi_abc[1];
        ellipsoid._c = voi_abc[2];

        unsigned int begin[3] , end[3];
        return ArithmeticUtils::get_valid_region(volume_data->_dim , ellipsoid , begin , end);
    }
}

void NoduleAnnotation::slot_open_nodule_file_i()
{
    if (!_is_ready)
    {
        return;
    }

    if (!_model_voi->get_vois().empty())
    {
        if (QMessageBox::No == QMessageBox::warning(
            this , tr("Load Nodule") , tr("You had annotated some of nodule . Will you discard them and load a new nodule file"),
            QMessageBox::Yes | QMessageBox::No))
        {
            return;
        }
    }

    QString file_name = QFileDialog::getOpenFileName(this, tr("Load Nodule") , 
        (NoduleAnnoConfig::instance()->get_last_open_direction() + "/" +_volume_infos->get_data_header()->series_uid).c_str(), 
        tr("NoduleSet(*.csv);;NoduleSet(*.nraw)"));
    if (!file_name.isEmpty())
    {
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        NoduleSetParser parser;
        parser.set_series_id(_volume_infos->get_data_header()->series_uid);
        std::string file_name_std(file_name.toLocal8Bit());

        IOStatus status;
        const bool is_csv = file_name_std.substr(file_name_std.size() - 3 , 3) == std::string("csv");
        if (!is_csv)
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
            _model_voi->notify(VOIModel::DELETE_VOI);

            const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
            int idx = 0;
            for (auto it = vois.begin() ; it != vois.end() ; ++it)
            {
                if (0 != _check_nodule_valid(_volume_infos->get_volume() , _volume_infos->get_camera_calculator() , *it))
                {
                    //TODO warning
                    std::stringstream ss;
                    ss << "The nodule item : center " << (*it).center.x << " " << (*it).center.y << " " <<
                        (*it).center.z << " , diameter " << (*it).diameter << " bounding overflow. So skip it.";
                    QMessageBox::warning(this ,tr("Load Nodule"), tr(ss.str().c_str()), QMessageBox::Yes);
                }
                else
                {
                    _model_voi->add_voi(*it , MaskLabelStore::instance()->acquire_label());
                    _model_voi->notify(VOIModel::ADD_VOI);
                }
            }

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
    //MI_NODULEANNO_LOG(MI_DEBUG) << "CellSelect "<< row << " " << column;
    VOISphere voi = _model_voi->get_voi(row);
    const Matrix4 mat_p2w = _mpr_scene_00->get_camera_calculator()->get_patient_to_world_matrix();
    _model_crosshair->locate(mat_p2w.transform(voi.center),  false);
    _model_crosshair->notify();
    _select_voi_id = row;
    
    if (this->_current_operation.compare("Finetune") == 0)
    {
        this->shift_tune_object();
    }
}

void NoduleAnnotation::slot_voi_table_widget_item_changed_i(QTableWidgetItem *item)
{
    const int row = item->row();
    const int column = item->column();
    if (1 == column)
    {
        std::string sDiameter =  (item->text()).toLocal8Bit();
        StrNumConverter<double> con;
        _model_voi->modify_diameter(row , con.to_num(sDiameter));
        _ob_scene_container->update();
    }
}

void NoduleAnnotation::slot_add_nodule_i()
{
    refresh_nodule_list_i();
}

void NoduleAnnotation::slot_delete_nodule_i()
{
    if (_select_voi_id >= 0 && _select_voi_id < _model_voi->get_vois().size())
    {
        _model_voi->remove_voi(_select_voi_id);
        _model_voi->notify(VOIModel::DELETE_VOI);
        _select_voi_id = -1;
        _ob_scene_container->update();
    }

    if (this->_current_operation.compare("Finetune") == 0)
    {
        this->shift_tune_object();
    }
}

void NoduleAnnotation::slot_voi_table_widget_nodule_type_changed_i(int id)
{
    QWidget* widget = _ui.tableWidgetNoduleList->cellWidget(id , 2);

    QComboBox* pBox= dynamic_cast<QComboBox*>(widget);
    if (pBox)
    {
        std::string type = pBox->currentText().toStdString();
        _model_voi->modify_name(id , type);
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
        NoduleAnnoConfig::instance()->get_preset_wl(CT_LUNGS , ww , wl);
    }
    else if (wl_preset == std::string("CT_Chest"))
    {
        NoduleAnnoConfig::instance()->get_preset_wl(CT_CHEST , ww , wl);
    }
    else if (wl_preset == std::string("CT_Bone"))
    {
        NoduleAnnoConfig::instance()->get_preset_wl(CT_BONE , ww , wl);
    }
    else if (wl_preset == std::string("CT_Angio"))
    {
        NoduleAnnoConfig::instance()->get_preset_wl(CT_ANGIO , ww , wl);
    }
    else if (wl_preset == std::string("CT_Abdomen"))
    {
        NoduleAnnoConfig::instance()->get_preset_wl(CT_ABDOMEN , ww , wl);
    }
    else if (wl_preset == std::string("CT_Brain"))
    {
        NoduleAnnoConfig::instance()->get_preset_wl(CT_BRAIN , ww , wl);
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

void NoduleAnnotation::change_layout1x1_i(int scan_type)
{
    SceneContainer* target_container = nullptr;
    QScrollBar* target_scroll_bar = nullptr;

    if (scan_type == static_cast<int>(SAGITTAL))
    {
        target_container = _mpr_00;
        target_scroll_bar = _mpr_00_scroll_bar;
    }
    else if (scan_type == static_cast<int>(CORONAL))
    {
        target_container = _mpr_01;
        target_scroll_bar = _mpr_01_scroll_bar;
    }
    else if (scan_type == static_cast<int>(TRANSVERSE))
    {
        target_container = _mpr_10;
        target_scroll_bar = _mpr_10_scroll_bar;
    }
    else
    {
        return;
    }

    if (0 == _layout_tag)
    {
        save_layout2x2_parameter_i();
    }

    _mpr_00->hide();
    _mpr_00_scroll_bar->hide();
    _mpr_01->hide();
    _mpr_01_scroll_bar->hide();
    _mpr_10->hide();
    _mpr_10_scroll_bar->hide();
    _vr_11->hide();

    _ui.gridLayout_6->removeWidget(_mpr_00);
    _ui.gridLayout_6->removeWidget(_mpr_00_scroll_bar);
    _ui.gridLayout_6->removeWidget(_mpr_01);
    _ui.gridLayout_6->removeWidget(_mpr_01_scroll_bar);
    _ui.gridLayout_6->removeWidget(_mpr_10 );
    _ui.gridLayout_6->removeWidget(_mpr_10_scroll_bar);
    _ui.gridLayout_6->removeWidget(_vr_11);

    _ui.gridLayout_6->addWidget(target_container , 0 ,0);
    _ui.gridLayout_6->addWidget(target_scroll_bar , 0 ,1,1,1);

    target_container->show();
    target_scroll_bar->show();
    target_container->update_scene();

    _layout_tag = 1;
}

void NoduleAnnotation::slot_scene_min_max_hint_i(const std::string& name)
{
    if (!_is_ready)
    {
        return;
    }

    if (0 == _layout_tag)
    {
        if (name == _mpr_scene_00->get_name())
        {
            change_layout1x1_i(SAGITTAL);
        }
        else if (name == _mpr_scene_01->get_name())
        {
            change_layout1x1_i(CORONAL);
        }
        else if (name == _mpr_scene_10->get_name())
        {
            change_layout1x1_i(TRANSVERSE);
        }
        else
        {
            return;
        }
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
        _model_focus->set_focus_scene_container(_mpr_00);
    }
    else if (name == _mpr_scene_01->get_name())
    {
        _model_focus->set_focus_scene_container(_mpr_01);
    }
    else if (name == _mpr_scene_10->get_name())
    {
        _model_focus->set_focus_scene_container(_mpr_10);
    }
    else
    {

    }
}

//void NoduleAnnotation::slot_focus_out_scene_i(QString name)
//{
//    if (!_is_ready)
//    {
//        return;
//    }
//}

void NoduleAnnotation::slot_crosshair_visibility_i(int iFlag)
{
    if (!_is_ready)
    {
        return;
    }

    _model_crosshair->set_visibility(iFlag != 0);

    SceneContainer* containers[3] = {_mpr_00 , _mpr_01 , _mpr_10};
    std::shared_ptr<MPRScene> scenes[3] = {_mpr_scene_00 , _mpr_scene_01 , _mpr_scene_10};
    if (0 == iFlag)//Hide
    {
        if (_ui.pushButtonArrow->isChecked())
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
        if (_ui.pushButtonArrow->isChecked())
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

void NoduleAnnotation::slot_nodule_overlay_visibility_i(int flag)
{
    if (!_is_ready)
    {
        return;
    }

    std::shared_ptr<MPRScene> scenes[3] = {_mpr_scene_00 , _mpr_scene_01 , _mpr_scene_10};
    MaskOverlayMode mode = flag != 0 ? MASK_OVERLAY_ENABLE : MASK_OVERLAY_DISABLE;
    for(int i= 0 ; i < 3 ; ++i)
    {
        scenes[i]->set_mask_overlay_mode(mode);
        scenes[i]->set_dirty(true);
    }
    _ob_scene_container->update();
}

void NoduleAnnotation::refresh_nodule_list_i()
{
    //reset nodule list
    _ui.tableWidgetNoduleList->clear();
    _ui.tableWidgetNoduleList->setRowCount(0);

    //set column header
    QTableWidgetItem *qtablewidgetitem = new QTableWidgetItem();
    QTableWidgetItem *qtablewidgetitem1 = new QTableWidgetItem();
    QTableWidgetItem *qtablewidgetitem2 = new QTableWidgetItem();

    qtablewidgetitem->setText(QApplication::translate("NoduleAnnotationClass", "Position", 0, QApplication::UnicodeUTF8));
    qtablewidgetitem1->setText(QApplication::translate("NoduleAnnotationClass", "Diameter", 0, QApplication::UnicodeUTF8));
    qtablewidgetitem2->setText(QApplication::translate("NoduleAnnotationClass", "Type", 0, QApplication::UnicodeUTF8));

    _ui.tableWidgetNoduleList->setHorizontalHeaderItem(0, qtablewidgetitem);
    _ui.tableWidgetNoduleList->setHorizontalHeaderItem(1, qtablewidgetitem1);
    _ui.tableWidgetNoduleList->setHorizontalHeaderItem(2, qtablewidgetitem2);

    //reset selected voi id
    _select_voi_id = -1;
    const std::vector<VOISphere>& vois = _model_voi->get_vois();
    if (!vois.empty())
    {
        _ui.tableWidgetNoduleList->setRowCount(vois.size());//Set row count , otherwise set item useless
        StrNumConverter<double> converter;
        const int iPrecision = 1;
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
            _ui.tableWidgetNoduleList->setItem(iRow,0, pPos);
            _ui.tableWidgetNoduleList->setItem(iRow,1, new QTableWidgetItem(sRadius.c_str()));

            QComboBox * pNoduleType = new QComboBox(_ui.tableWidgetNoduleList);
            pNoduleType->clear();
            for (int i = 0 ; i<NODULE_TYPE_NUM ; ++i)
            {
                pNoduleType->insertItem(i , S_NODULE_TYPES[i].c_str());
                pNoduleType->setItemData(i , S_NODULE_TYPE_DESCRIPTION[i].c_str(), Qt::ToolTipRole);
            }
            _ui.tableWidgetNoduleList->setCellWidget(iRow,2, pNoduleType);

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

void NoduleAnnotation::closeEvent(QCloseEvent * event)
{
    release_i();

    GLTextureCache::instance()->process_cache();
    GLResourceManagerContainer::instance()->update_all();
    MI_NODULEANNO_LOG(MI_DEBUG) << "GL context status: " << GLContextHelper::has_gl_context();

    NoduleAnnoConfig::instance()->finalize();

    QMainWindow::closeEvent(event);
}

void NoduleAnnotation::slot_quit_i()
{
    this->close();

}

void NoduleAnnotation::slot_save_label_file()
{
    //TODO: check whether data are ready

    // label remapping
    const auto & labels = this->_model_voi->get_labels(); // get the labels which are ordered

    // build a map to 'correct' the labels to be sequential, starting from 1
    std::map<unsigned char, unsigned char> label_correction;

    unsigned char label_index = 1;
    for (auto it = labels.begin(); it != labels.end(); ++it, ++label_index)
    {
        label_correction[(*it)] = label_index;
        //MI_NODULEANNO_LOG(MI_DEBUG)<< static_cast<int>(*it) << " mapped to " << static_cast<int>(cnt);
    }

    // create a new data as the final output
    ImageData * output_label_volume = new ImageData();
    std::shared_ptr<ImageData> mask_data = this->_volume_infos->get_mask(); //get the associated mask
    mask_data->deep_copy(output_label_volume);

    // iterate/correct the mask data as 1D array
    unsigned char* array_pointer = static_cast<unsigned char*>(output_label_volume->get_pixel_pointer());
    if (array_pointer == nullptr)
    {
        MI_NODULEANNO_LOG(MI_ERROR) << "out label volume memory is null.";
        return;
    }

    unsigned int total_number_of_voxels = mask_data->_dim[0] * mask_data->_dim[1] * mask_data->_dim[2];
    for (int voxel=0; voxel < total_number_of_voxels; ++voxel)
    {
        if (array_pointer[voxel] != 0) // skip empty voxels
        {
            auto it = label_correction.find(array_pointer[voxel]);
            if (it != label_correction.end())
            {
                array_pointer[voxel] = it->second;
            }
        }
    }


    // encode the original labels
    std::vector<unsigned int> run_length_encoded_output = RunLengthOperator::encode(array_pointer, total_number_of_voxels);
    delete output_label_volume;

    // write to disk
    QString output_file_name = QFileDialog::getSaveFileName(
        this,
        tr("Save Label"), 
        QString(this->_volume_infos->get_data_header()->series_uid.c_str()), tr("LabelSet(*.rle)") );

    if (!output_file_name.isEmpty())
    {
        std::string output_file_name_std(output_file_name.toLocal8Bit());

        std::ofstream output_file(output_file_name_std, std::ios::out | std::ios::binary);
        if (output_file.is_open())
        {
            unsigned int * raw_ptr = run_length_encoded_output.data();
            output_file.write(
                reinterpret_cast<char *>(raw_ptr),
                sizeof(unsigned int) * run_length_encoded_output.size());
            output_file.flush();
            output_file.close();
        }
    }
}

void NoduleAnnotation::update_last_open_direction_i(const std::string& file_path)
{
    int sub_file_path = -1;
    for(int i = file_path.size() - 1 ; i >= 0 ;--i)
    {
        if(file_path[i] == '/' || file_path[i] == '\\')
        {
            sub_file_path = i;
            const std::string last_open_direction = file_path.substr(0 , sub_file_path);
            NoduleAnnoConfig::instance()->set_last_open_direction(last_open_direction);
            break;
        }
    }
}

//
//void NoduleAnnotation::write_encoded_labels(std::string& file_name, std::vector<unsigned int> &run_length_encoded_output)
//{
//    QString output_file_name = QFileDialog::getSaveFileName(
//        this,
//        tr("Save Label"), 
//        QString(this->_volume_infos->get_data_header()->series_uid.c_str()), tr("LabelSet(*.rle)") );
//
//    if (!output_file_name.isEmpty())
//    {
//        std::string output_file_name_std(output_file_name.toLocal8Bit());
//
//        bool binary_output = true;
//        if (binary_output)
//        {
//            std::ofstream output_file(output_file_name_std, std::ios::out | std::ios::binary);
//            if (output_file.is_open())
//            {
//                int * raw_ptr = run_length_encoded_output.data();
//                output_file.write(
//                    reinterpret_cast<char *>(raw_ptr),
//                    sizeof(int) * run_length_encoded_output.size());
//                output_file.flush();
//                output_file.close();
//            }
//        }
//        else /*for debug*/
//        {
//            std::ofstream output_file;
//            output_file.open(output_file_name_std, std::ios::out);
//            if (output_file.is_open())
//            {
//                std::ostream_iterator<int> out_it (output_file, ", ");
//                std::copy ( run_length_encoded_output.begin(), run_length_encoded_output.end(), out_it );
//                output_file.flush();
//                output_file.close();
//            }
//        }
//    }
//}

void NoduleAnnotation::slot_load_label_file()
{
    if (!_is_ready || !this->_volume_infos)
    {
        MI_NODULEANNO_LOG(MI_WARNING) << "No volume loaded yet.";
        return;
    }

    if (!_model_voi->get_vois().empty())
    {
        if (QMessageBox::No == QMessageBox::warning(
            this , tr("Load Nodule") , tr("You had annotated some of nodule . Will you discard them and load a new label file"),
            QMessageBox::Yes | QMessageBox::No))
        {
            return;
        }
    }

    // read file, interpret as pairs of integers, (repeated times, value)
    QString label_file_name_str = QFileDialog::getOpenFileName(this, tr("Load Label") , 
        (NoduleAnnoConfig::instance()->get_last_open_direction() + "/" +_volume_infos->get_data_header()->series_uid).c_str(),
        tr("LabelSet(*.rle)"));
    if (label_file_name_str.isEmpty())
    {
        return;
    }
    std::string file_name_std(label_file_name_str.toLocal8Bit());
    std::ifstream input_file(file_name_std, std::ios::in | std::ios::binary | std::ios::ate);
    if (!input_file.is_open())
    {
        return;
    }

    // get size in bytes
    input_file.seekg (0, input_file.end);
    int file_size = input_file.tellg();
    input_file.seekg (0, input_file.beg);

    // prepare the buffer and copy into it
    int number_of_entries = file_size/sizeof(int);
    std::vector<unsigned int> labels(number_of_entries);
    char * buffer = reinterpret_cast<char*>(labels.data());
    input_file.read (buffer, file_size);

    //if (input_file)
    //    MI_NODULEANNO_LOG(MI_DEBUG) << "all characters read successfully.";
    //else
    //    MI_NODULEANNO_LOG(MI_DEBUG) << "error: only " << input_file.gcount() << " could be read";

    input_file.close();

    // count the voxels, check w.r.t _volume_infos.dim[0]*dim[1]*dim[2]
    unsigned int sum_voxels = 0;
    for (auto it = labels.begin(); it != labels.end(); it += 2)
    {
        sum_voxels += (*it);
    }
    MI_NODULEANNO_LOG(MI_DEBUG) << sum_voxels << " labels are loaded.";

    std::shared_ptr<ImageData> mask_in_use = this->_volume_infos->get_mask();
    unsigned int total_number_of_voxels = mask_in_use->_dim[0] * mask_in_use->_dim[1] * mask_in_use->_dim[2];
    if (sum_voxels != total_number_of_voxels)
    {
        MI_NODULEANNO_LOG(MI_ERROR) << "Fail since label data not match the volume in use.";
        return;
    }

    // delete previous files
    this->_model_voi->remove_all();
    this->_model_voi->notify(VOIModel::DELETE_VOI);

    // TODO: directly update the mask_data? but if in the presence of existing vois (i.e., spheres), how to combine them?
    //clock_t _start = clock();
    std::vector<unsigned char> actual_labels = RunLengthOperator::decode(labels);
    /*clock_t _end = clock();
    MI_NODULEANNO_LOG(MI_DEBUG) << "decode cost : " << double(_end - _start) << " ms.";*/

    std::shared_ptr<ImageData> volume = _volume_infos->get_volume(); 
    double origin[3] = {volume->_image_position.x, volume->_image_position.y, volume->_image_position.z}; // TODO here we ignore rotation
    std::vector<VOISphere> voi_spheres = MaskVOIConverter::convert_label_2_sphere(actual_labels, volume->_dim, volume->_spacing, origin);
    const Matrix4 matv2patient = _volume_infos->get_camera_calculator()->get_world_to_patient_matrix()*_volume_infos->get_camera_calculator()->get_volume_to_world_matrix();
    for (auto it = voi_spheres.begin() ; it != voi_spheres.end() ; ++it)
    {
        (*it).center = matv2patient.transform((*it).center);
        if ((*it).diameter <DOUBLE_EPSILON)
        {
            (*it).diameter = (std::min)((std::min)(volume->_spacing[0] , volume->_spacing[1]) , volume->_spacing[2]);
        }
    }

    //clock_t _end2 = clock();
    //MI_NODULEANNO_LOG(MI_DEBUG) << "convert label cost : " << double(_end2 - _end) << " ms.";

    // update underlying mask
    bool directly_update_mask = true;
    if (directly_update_mask)
    {
        unsigned char * dest_value_ptr = static_cast<unsigned char *>(mask_in_use->get_pixel_pointer());
        unsigned char * src_value = new unsigned char [total_number_of_voxels];

        for (int voxel=0; voxel<total_number_of_voxels;++voxel)
        {
            unsigned int current_label = actual_labels.at(voxel);

            // populate the temporary information container
            src_value[voxel] = current_label;

            // update current volume with loaded labels
            dest_value_ptr[voxel] = current_label;
        }

        // Notify that mask volume has changed, renderer should refresh GPU memory [lower_bound, upper_bound) // half-open interval
        unsigned int lower_bound[3] = {0,0,0};
        unsigned int upper_bound[3] = {mask_in_use->_dim[0], mask_in_use->_dim[1], mask_in_use->_dim[2]};

        if (Configure::instance()->get_processing_unit_type() == GPU)
        {
            this->_volume_infos->update_mask(lower_bound, upper_bound, src_value, true); // "src_value": temporary data container; "true" : already update loaded
        }
    }

    // update the underlying model & visual representation
    for (int sphere_idx=0; sphere_idx<voi_spheres.size(); ++sphere_idx)
    {
        this->_model_voi->add_voi(voi_spheres.at(sphere_idx), sphere_idx+1);
    }
    this->_model_voi->notify(VOIModel::LOAD_VOI);
    
    QMessageBox::information(this , tr("Load Label") , tr("Load label file success."),QMessageBox::Ok);
} 