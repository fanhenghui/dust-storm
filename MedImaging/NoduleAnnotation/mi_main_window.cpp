#include "mi_main_window.h"

#include <iostream>
#include <fstream>
#include <sstream>

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
#include "MedImgIO/mi_run_length_operator.h"

#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_mask_label_store.h"

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
#include "MedImgQtWidgets/mi_mouse_op_test.h"
#include "MedImgQtWidgets/mi_model_voi.h"
#include "MedImgQtWidgets/mi_model_cross_hair.h"
#include "MedImgQtWidgets/mi_model_focus.h"
#include "MedImgQtWidgets/mi_observer_scene_container.h"
#include "MedImgQtWidgets/mi_observer_progress.h"
#include "MedImgQtWidgets/mi_observer_voi_statistic.h"
#include "MedImgQtWidgets/mi_observer_voi_segment.h"


#include "mi_observer_voi_table.h"
#include "mi_observer_mpr_scroll_bar.h"
#include "mi_mouse_op_min_max_hint.h"
#include "mi_my_rsa.h"
#include "mi_dicom_anonymization_dialog.h"
#include "mi_raw_data_import_dialog.h"

#include <QEvent>
#include <QSizePolicy>
#include <QScrollBar>
#include <QFileDialog>
#include <QMessagebox>
#include <QSignalMapper>
#include <QProgressDialog>

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
static const std::string S_NODULE_TYPE_DESCRIPTION_CHINESE[NODULE_TYPE_NUM] = 
{
    "边界清晰的结节",
    "粘连血管的结节",
    "肺壁游离的结节",
    "粘连肺壁的结节",
    "毛玻璃型结节",
    "非结节"
};

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
    _ui.setupUi(this);

    _ui.tableWidgetNoduleList->setSelectionBehavior(QAbstractItemView::SelectRows);
    _ui.tableWidgetNoduleList->setSelectionMode(QAbstractItemView::SingleSelection);

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
    //1 TODO Check process unit
    //Open config file
    std::fstream input_file("../../../config/configure.txt" , std::ios::in);
    if (!input_file.is_open())
    {
        input_file.open("./config/configure.txt" , std::ios::in);//second chance
    }

    if (!input_file.is_open())
    {
        Configuration::instance()->set_processing_unit_type(GPU);
        Configuration::instance()->set_nodule_file_rsa(true);
    }
    else
    {
        std::string line;
        std::string tag;
        std::string equal;
        std::string context;
        while(std::getline(input_file,line))
        {
            std::stringstream ss(line);
            ss >> tag >> equal >> context;
            if (tag == std::string("ProcessingUnit"))
            {
                if (context == "GPU")
                {
                    Configuration::instance()->set_processing_unit_type(GPU);
                }
                else
                {
                    Configuration::instance()->set_processing_unit_type(CPU);
                }
            }

            if (tag == "NoduleOutput")
            {
                if(context == "TEXT")
                {
                    Configuration::instance()->set_nodule_file_rsa(false);
                }
                else
                {
                    Configuration::instance()->set_nodule_file_rsa(true);
                }
            }
        }
        input_file.close();
    }


    GLUtils::set_check_gl_flag(false);
}

void NoduleAnnotation::create_scene_i()
{
    _mpr_scene_00.reset(new MPRScene(_mpr_00->width() , _mpr_00->height()));
    _mpr_scene_01.reset(new MPRScene(_mpr_01->width() , _mpr_01->height()));
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
        graphic_item_mpr_border->set_focus_model(_model_focus);
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

    //File
    connect(_ui.actionOpen_DICOM_Folder , SIGNAL(triggered()) , this , SLOT(slot_open_dicom_folder_i()));
    connect(_ui.actionOpen_Meta_Image , SIGNAL(triggered()) , this , SLOT(slot_open_meta_image_i()));
    connect(_ui.actionOpen_Raw , SIGNAL(triggered()) , this , SLOT(slot_open_raw_i()));
    connect(_ui.actionSave_Nodule , SIGNAL(triggered()) , this , SLOT(slot_save_nodule_file_i()));
    connect(_ui.actionLoad_Nodule , SIGNAL(triggered()) , this , SLOT(slot_open_nodule_file_i()));
    connect(_ui.actionAnonymization_DICOM , SIGNAL(triggered()) , this , SLOT(slot_dicom_anonymization_i()));
    connect(_ui.actionQuit , SIGNAL(triggered()) , this , SLOT(slot_quit_i()));

    connect(_ui.actionSave_Label, SIGNAL(triggered()) , this , SLOT(slot_save_label_file()));
    //connect(_ui.actionLoad_Label, SIGNAL(triggered()) , this , SLOT(slot_load_label_file()));

    //MPR scroll bar
    connect(_mpr_00_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr00_i(int)));
    connect(_mpr_01_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr01_i(int)));
    connect(_mpr_10_scroll_bar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr10_i(int)));

    //Common tools
    connect(_ui.pushButtonArrow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_arrow_i()));
    connect(_ui.pushButtonAnnotate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_annotate_i()));
    connect(_ui.pushButtonRotate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_rotate_i()));
    connect(_ui.pushButtonZoom , SIGNAL(pressed()) , this , SLOT(slot_press_btn_zoom_i()));
    connect(_ui.pushButtonPan , SIGNAL(pressed()) , this , SLOT(slot_press_btn_pan_i()));
    connect(_ui.pushButtonWindowing , SIGNAL(pressed()) , this , SLOT(slot_press_btn_windowing_i()));
    connect(_ui.pushButtonFitWindow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_fit_window_i()));

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

void NoduleAnnotation::slot_open_dicom_folder_i()
{
    QStringList file_name_list = QFileDialog::getOpenFileNames(
        this ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    if (!file_name_list.empty())
    {
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
    _select_vio_id = -1;

    _is_ready = true;
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
        this ,tr("Loading meta data"),"",tr("Dicom image(*mhd)"));

    if (file_name.isEmpty())
    {
        return;
    }

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
    mpr_containers.push_back(_mpr_10);

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
    mpr_containers.push_back(_mpr_10);

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
    mpr_containers.push_back(_mpr_10);

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
    mpr_containers.push_back(_mpr_10);

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
    mpr_containers.push_back(_mpr_10);

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

    QString file_name = QFileDialog::getSaveFileName(this, tr("Save Nodule") , QString(_volume_infos->get_data_header()->series_uid.c_str()), tr("NoduleSet(*.csv)"));
    if (!file_name.isEmpty())
    {
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        const std::vector<VOISphere>& vois = _model_voi->get_vois();
        nodule_set->set_nodule(vois);

        NoduleSetParser parser;
        parser.set_series_id(_volume_infos->get_data_header()->series_uid);
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

    if (!_model_voi->get_vois().empty())
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
        parser.set_series_id(_volume_infos->get_data_header()->series_uid);
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
            _model_voi->notify(VOIModel::DELETE_VOI);

            const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
            int idx = 0;
            for (auto it = vois.begin() ; it != vois.end() ; ++it)
            {
                _model_voi->add_voi(*it , MaskLabelStore::instance()->acquire_label());
                _model_voi->notify(VOIModel::ADD_VOI);
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
    //std::cout << "CellSelect "<< row << " " << column<< std::endl; 
    VOISphere voi = _model_voi->get_voi(row);
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
    if (_select_vio_id >= 0 && _select_vio_id < _model_voi->get_vois().size())
    {
        _model_voi->remove_voi(_select_vio_id);
        _model_voi->notify(VOIModel::DELETE_VOI);
        _select_vio_id = -1;
        _ob_scene_container->update();
    }
}

void NoduleAnnotation::slot_voi_table_widget_nodule_type_changed_i(int id)
{
    QWidget* widget = _ui.tableWidgetNoduleList->cellWidget(id , 2);

    QComboBox* pBox= dynamic_cast<QComboBox*>(widget);
    if (pBox)
    {
        std::string type = pBox->currentText().toStdString();
        std::cout << id <<'\t' << type << std::endl;

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
            target_container = _mpr_10;
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
    _select_vio_id = -1;
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
        //std::cout << static_cast<int>(*it) << " mapped to " << static_cast<int>(cnt) << '\n';
    }

    // create a new data as the final output
    ImageData * output_label_volume = new ImageData();
    std::shared_ptr<ImageData> mask_data = this->_volume_infos->get_mask(); //get the associated mask
    mask_data->deep_copy(output_label_volume);

    // iterate/correct the mask data as 1D array
    unsigned char* array_pointer = static_cast<unsigned char*>(output_label_volume->get_pixel_pointer());
    if (array_pointer == nullptr)
    {
        std::cout << "We get a null pointer for the label :( \n";
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
    std::vector<unsigned int> run_length_encoded_output = RunLengthOperator::Encode(array_pointer, total_number_of_voxels);
    delete output_label_volume;

    //std::vector<int> run_length_encoded_output = this->RunLengthEncodeLabel(label_correction);

    // write to disk
    QString output_file_name = QFileDialog::getSaveFileName(
        this,
        tr("Save Label"), 
        QString(this->_volume_infos->get_data_header()->series_uid.c_str()), tr("LabelSet(*.rle)") );

    if (!output_file_name.isEmpty())
    {
        std::string output_file_name_std(output_file_name.toLocal8Bit());
        this->write_encoded_labels(output_file_name_std, run_length_encoded_output);
    }
    // this->WriteEncodedLabels(run_length_encoded_output);    
}

void NoduleAnnotation::write_encoded_labels(std::string& file_name, std::vector<unsigned int> &run_length_encoded_output)
{
    //QString output_file_name = QFileDialog::getSaveFileName(
    //    this,
    //    tr("Save Label"), 
    //    QString(this->_volume_infos->get_data_header()->series_uid.c_str()), tr("LabelSet(*.rle)") );

    //if (!output_file_name.isEmpty())
    //{
    //    std::string output_file_name_std(output_file_name.toLocal8Bit());

    //    bool binary_output = true;
    //    if (binary_output)
    //    {
    //        std::ofstream output_file(output_file_name_std, std::ios::out | std::ios::binary);
    //        if (output_file.is_open())
    //        {
    //            int * raw_ptr = run_length_encoded_output.data();
    //            output_file.write(
    //                reinterpret_cast<char *>(raw_ptr),
    //                sizeof(int) * run_length_encoded_output.size());
    //            output_file.flush();
    //            output_file.close();
    //        }
    //    }
    //    else /*for debug*/
    //    {
    //        std::ofstream output_file;
    //        output_file.open(output_file_name_std, std::ios::out);
    //        if (output_file.is_open())
    //        {
    //            std::ostream_iterator<int> out_it (output_file, ", ");
    //            std::copy ( run_length_encoded_output.begin(), run_length_encoded_output.end(), out_it );
    //            output_file.flush();
    //            output_file.close();
    //        }
    //    }
    //}


    std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
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

//void NoduleAnnotation::slot_load_label_file()
//{
//    //if (!this->_volume_infos)
//    //{
//    //    std::cout << "No volume loaded yet!\n";
//    //    return;
//    //}
//
//    // read file, interpret as pairs of integers, (repeated times, value)
//    QString label_file_name_str = QFileDialog::getOpenFileName(this, tr("Load Label") , "home/" /*QString(_volume_infos->get_data_header()->series_uid.c_str())*/, tr("LabelSet(*.raw)"));
//    if (label_file_name_str.isEmpty())
//    {
//        return;
//    }
//    std::string file_name_std(label_file_name_str.toLocal8Bit());
//    std::ifstream input_file(file_name_std, std::ios::in | std::ios::binary | std::ios::ate);
//    if (!input_file.is_open())
//    {
//        return;
//    }
//
//    // get size in bytes
//    input_file.seekg (0, input_file.end);
//    int file_size = input_file.tellg();
//    input_file.seekg (0, input_file.beg);
//
//    // prepare the buffer and copy into it
//    int number_of_entries = file_size/sizeof(int);
//    std::vector<int> labels(number_of_entries);
//    char * buffer = reinterpret_cast<char*>(labels.data());
//    input_file.read (buffer, file_size);
//
//    //if (input_file)
//    //    std::cout << "all characters read successfully.";
//    //else
//    //    std::cout << "error: only " << input_file.gcount() << " could be read";
//
//    input_file.close();
//
//    // count the voxels, check w.r.t _volume_infos.dim[0]*dim[1]*dim[2]
//    unsigned int sum_voxels = 0;
//    for (auto it = labels.begin(); it != labels.end(); it += 2)
//    {
//        sum_voxels += (*it);
//    }
//    std::cout << sum_voxels << " labels are loaded\n";
//
//    std::shared_ptr<ImageData> mask_in_use = this->_volume_infos->get_mask();
//    unsigned int total_number_of_voxels = mask_in_use->_dim[0] * mask_in_use->_dim[1] * mask_in_use->_dim[2];
//    if (sum_voxels != total_number_of_voxels)
//    {
//        std::cout << "label data FAILED to match the mask in use \n";
//        return;
//    }
//
//    // TODO: directly update the mask_data? but if in the presence of existing vois (i.e., spheres), how to combine them?
//    bool directly_update_mask = true;
//    std::vector<unsigned char> unique_labels;
//    unsigned char * dest_value_ptr = static_cast<unsigned char *>(mask_in_use->get_pixel_pointer());
//    unsigned char * src_value = new unsigned char [total_number_of_voxels];
//
//    if (directly_update_mask)
//    {
//        unsigned int current_index = 0;
//        unsigned char current_label = static_cast<unsigned char>( labels[current_index+1] );
//
//        if (current_label != 0)
//            unique_labels.push_back(current_label);
//
//        for (int voxel=0; voxel<total_number_of_voxels;++voxel)
//        {
//            if (voxel >= labels[current_index] )
//            {
//                current_index += 2;
//                labels[current_index] += labels[current_index-2];
//                current_label = static_cast<unsigned char>( labels[current_index+1] );
//
//                if ( current_label != 0 && std::find(unique_labels.begin(), unique_labels.end(), current_label) == unique_labels.end() )
//                    unique_labels.push_back(current_label);
//            }
//
//            // populate the temporary information container
//            src_value[voxel] = current_label;
//
//            // update current volume with loaded labels
//            dest_value_ptr[voxel] = current_label;
//
//            //// on-line debug: write out, then read in, and compare!
//            //if (dest_value_ptr[voxel] != current_label)
//            //{
//            //    std::cout << "Voxel " << voxel <<" has wrong value!\n";
//            //}
//        }
//    }
//
//    // Notify that mask volume has changed, renderer should refresh GPU memory
//    // [lower_bound, upper_bound)
//    unsigned int lower_bound[3] = {0,0,0};
//    unsigned int upper_bound[3] = {mask_in_use->_dim[0], mask_in_use->_dim[1], mask_in_use->_dim[2]};
//
//    this->_volume_infos->update_mask(lower_bound, upper_bound, src_value, true); // "src_value": temporary data container; "true" : already update loaded
//    this->_ob_scene_container->update();
//}