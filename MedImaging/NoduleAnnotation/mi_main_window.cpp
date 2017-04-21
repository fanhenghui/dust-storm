#include "mi_main_window.h"

#include <iostream>

#include "MedImgCommon/mi_concurrency.h"
#include "MedImgCommon/mi_configuration.h"
#include "MedImgCommon/mi_string_number_converter.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgIO/mi_meta_object_loader.h"
#include "MedImgIO/mi_nodule_set.h"
#include "MedImgIO/mi_nodule_set_csv_parser.h"

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
#include "MedImgQtWidgets/mi_painter_corners_info.h"
#include "MedImgQtWidgets/mi_painter_voi.h"
#include "MedImgQtWidgets/mi_painter_cross_hair.h"
#include "MedImgQtWidgets/mi_painter_mpr_border.h"
#include "MedImgQtWidgets/mi_mouse_op_zoom.h"
#include "MedImgQtWidgets/mi_mouse_op_pan.h"
#include "MedImgQtWidgets/mi_mouse_op_rotate.h"
#include "MedImgQtWidgets/mi_mouse_op_mpr_paging.h"
#include "MedImgQtWidgets/mi_mouse_op_windowing.h"
#include "MedImgQtWidgets/mi_mouse_op_probe.h"
#include "MedImgQtWidgets/mi_mouse_op_annotate.h"
#include "MedImgQtWidgets/mi_mouse_op_locate.h"
#include "MedImgQtWidgets/mi_model_voi.h"
#include "MedImgQtWidgets/mi_model_cross_hair.h"
#include "MedImgQtWidgets/mi_observer_scene_container.h"


#include "mi_observer_voi_table.h"
#include "mi_observer_mpr_scroll_bar.h"
#include "mi_mouse_op_min_max_hint.h"

#include "qevent.h"
#include "qsizepolicy.h"
#include "qscrollbar.h"
#include "qfiledialog.h"
#include "qmessagebox.h"
#include "qsignalmapper.h"

using namespace medical_imaging;

//Nodule type
const std::string ksNoduleTypeGGN = std::string("GGN");
const std::string ksNoduleTypeAAH = std::string("AAH");

//Preset WL
const float kfPresetCTAbdomenWW = 400;
const float kfPresetCTAbdomenWL = 60;

const float kfPresetCTLungsWW = 1500;
const float kfPresetCTLungsWL = -400;

const float kfPresetCTBrainWW = 80;
const float kfPresetCTBrainWL = 40;

const float kfPresetCTAngioWW = 600;
const float kfPresetCTAngioWL = 300;

const float kfPresetCTBoneWW = 1500;
const float kfPresetCTBoneWL = 300;

const float kfPresetCTChestWW = 400;
const float kfPresetCTChestWL = 40;

NoduleAnnotation::NoduleAnnotation(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags),
    m_iLayoutType(0),
    m_bReady(false),
    m_pNoduleObject(nullptr),
    m_pNoduleTypeSignalMapper(nullptr)
{
    ui.setupUi(this);

    ui.tableWidgetNoduleList->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui.tableWidgetNoduleList->setSelectionMode(QAbstractItemView::SingleSelection);

    m_pMPR00 = new SceneContainer(SharedWidget::instance());
    m_pMPR00->setMinimumSize(100,100);

    m_pMPR01 = new SceneContainer(SharedWidget::instance());
    m_pMPR01->setMinimumSize(100,100);

    m_pMPR10 = new SceneContainer(SharedWidget::instance());
    m_pMPR10->setMinimumSize(100,100);

    m_pVR11 = new SceneContainer(SharedWidget::instance());
    m_pVR11->setMinimumSize(100,100);

    m_pMPR00->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);//自适应窗口
    m_pMPR01->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);
    m_pMPR10->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);
    m_pVR11->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding);


    m_pMPR00ScrollBar = new QScrollBar(ui.centralWidget);
    m_pMPR00ScrollBar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR00"));
    m_pMPR00ScrollBar->setOrientation(Qt::Vertical);

    m_pMPR01ScrollBar = new QScrollBar(ui.centralWidget);
    m_pMPR01ScrollBar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR01"));
    m_pMPR01ScrollBar->setOrientation(Qt::Vertical);

    m_pMPR10ScrollBar = new QScrollBar(ui.centralWidget);
    m_pMPR10ScrollBar->setObjectName(QString::fromUtf8("verticalScrollBar_MPR10"));
    m_pMPR10ScrollBar->setOrientation(Qt::Vertical);


    ui.gridLayout_6->addWidget(m_pMPR00 , 0 ,0);
    ui.gridLayout_6->addWidget(m_pMPR00ScrollBar , 0 ,1,1,1);
    ui.gridLayout_6->addWidget(m_pMPR01 , 0 ,2);
    ui.gridLayout_6->addWidget(m_pMPR01ScrollBar , 0 ,3,1,1);
    ui.gridLayout_6->addWidget(m_pMPR10 , 1 ,0);
    ui.gridLayout_6->addWidget(m_pMPR10ScrollBar , 1 ,1,1,1);
    ui.gridLayout_6->addWidget(m_pVR11 , 1 ,2);

    m_pNoduleObject = new QNoduleObject(this);
    m_pMinMaxHintObject = new QMinMaxHintObject(this);

    configure_i();

    connectS_signal_slot_i();

}

NoduleAnnotation::~NoduleAnnotation()
{

}

void NoduleAnnotation::configure_i()
{
    //1 TODO Check process unit
    Configuration::instance()->set_processing_unit_type(GPU);
    GLUtils::set_check_gl_flag(false);
}

void NoduleAnnotation::create_scene_i()
{
    m_pMPRScene00.reset(new MPRScene(m_pMPR00->width() , m_pMPR00->height()));
    m_pMPRScene01.reset(new MPRScene(m_pMPR01->width() , m_pMPR01->height()));
    m_pMPRScene10.reset(new MPRScene(m_pMPR10->width() , m_pMPR10->height()));

    std::vector<MPRScenePtr> vecMPRScenes;
    vecMPRScenes.push_back(m_pMPRScene00);
    vecMPRScenes.push_back(m_pMPRScene01);
    vecMPRScenes.push_back(m_pMPRScene10);

    std::vector<SceneContainer*> vecMPRs;
    vecMPRs.push_back(m_pMPR00);
    vecMPRs.push_back(m_pMPR01);
    vecMPRs.push_back(m_pMPR10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        //1 Set Scene
        vecMPRs[i]->set_scene(vecMPRScenes[i]);

        //2 Set scene parameter
        vecMPRScenes[i]->set_volume_infos(m_pVolumeInfos);
        vecMPRScenes[i]->set_sample_rate(1.0);
        vecMPRScenes[i]->set_global_window_level(kfPresetCTLungsWW,kfPresetCTLungsWL);
        vecMPRScenes[i]->set_composite_mode(COMPOSITE_AVERAGE);
        vecMPRScenes[i]->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
        vecMPRScenes[i]->set_mask_mode(MASK_NONE);
        vecMPRScenes[i]->set_interpolation_mode(LINEAR);

        //3 Add painter list
        std::vector<std::shared_ptr<PainterBase>> vecPainterList;
        std::shared_ptr<CornersInfoPainter> pCornerInfo(new CornersInfoPainter());
        pCornerInfo->set_scene(vecMPRScenes[i]);
        vecPainterList.push_back(pCornerInfo);

        std::shared_ptr<VOIPainter> pVOI(new VOIPainter());
        pVOI->set_scene(vecMPRScenes[i]);
        pVOI->set_voi_model(m_pVOIModel);
        vecPainterList.push_back(pVOI);

        std::shared_ptr<CrosshairPainter> pCrosshair(new CrosshairPainter());
        pCrosshair->set_scene(vecMPRScenes[i]);
        pCrosshair->set_crosshair_model(m_pCrosshairModel);
        vecPainterList.push_back(pCrosshair);

        std::shared_ptr<MPRBorderPainter> pMPRBorder(new MPRBorderPainter());
        pMPRBorder->set_scene(vecMPRScenes[i]);
        pMPRBorder->set_crosshair_model(m_pCrosshairModel);
        vecPainterList.push_back(pMPRBorder);

        vecMPRs[i]->add_painter_list(vecPainterList);

        //4 Add operation 
        std::shared_ptr<MouseOpLocate> pMPRLocate(new MouseOpLocate());
        pMPRLocate->set_scene(vecMPRScenes[i]);
        pMPRLocate->set_crosshair_model(m_pCrosshairModel);

        std::shared_ptr<MouseOpMinMaxHint> pMinMaxHint(new MouseOpMinMaxHint());
        pMinMaxHint->set_scene(vecMPRScenes[i]);
        pMinMaxHint->set_min_max_hint_object(m_pMinMaxHintObject);

        IMouseOpPtrCollection vecOpsLBtn(2);
        vecOpsLBtn[0] = pMPRLocate;
        vecOpsLBtn[1] = pMinMaxHint;
        vecMPRs[i]->register_mouse_operation(vecOpsLBtn, Qt::LeftButton , Qt::NoModifier);

        std::shared_ptr<MouseOpZoom> pZoom(new MouseOpZoom());
        pZoom->set_scene(vecMPRScenes[i]);
        vecMPRs[i]->register_mouse_operation(pZoom , Qt::RightButton , Qt::NoModifier);

        /*std::shared_ptr<MouseOpWindowing> pWindowing(new MouseOpWindowing());
        pWindowing->set_scene(vecMPRScenes[i]);
        vecMPRs[i]->register_mouse_operation(pWindowing , Qt::MiddleButton , Qt::NoModifier);
*/
        std::shared_ptr<MouseOpPan> pPan(new MouseOpPan());
        pPan->set_scene(vecMPRScenes[i]);
        vecMPRs[i]->register_mouse_operation(pPan , Qt::LeftButton | Qt::RightButton , Qt::NoModifier);

    }

    //////////////////////////////////////////////////////////////////////////
    //Placement orthogonal MPR
    std::shared_ptr<CameraCalculator> pCameraCal = m_pVolumeInfos->get_camera_calculator();
    ScanSliceType aScanType[3] = {SAGITTAL ,CORONAL , TRANSVERSE};
    const std::string aScanTypeString[3] = {"Sagittal_MPR_scene_00" ,"Coronal_MPR_scene_01" , "Transverse_MPR_scene_10"};
    MPRScenePtr aScenes[3] = {m_pMPRScene00 , m_pMPRScene01 , m_pMPRScene10};
    RGBUnit aColors[3] ={kColorSagittal, kColorCoronal , kColorTransverse};
    QScrollBar* aScrollBar[3] = {m_pMPR00ScrollBar , m_pMPR01ScrollBar , m_pMPR10ScrollBar};

    //Model set scenes
    m_pCrosshairModel->set_mpr_scene(aScanType , aScenes , aColors);

    m_pMPRScrollBarOb->add_scroll_bar(m_pMPRScene00 , m_pMPR00ScrollBar);
    m_pMPRScrollBarOb->add_scroll_bar(m_pMPRScene01 , m_pMPR01ScrollBar);
    m_pMPRScrollBarOb->add_scroll_bar(m_pMPRScene10 , m_pMPR10ScrollBar);

    for (int i = 0 ; i<3 ; ++i)
    {
        aScenes[i]->place_mpr(aScanType[i]);
        aScenes[i]->set_name(aScanTypeString[i]);

        //Init page
        aScrollBar[i]->setMaximum(pCameraCal->get_page_maximum(aScanType[i])-1);
        aScrollBar[i]->setMinimum(0);
        aScrollBar[i]->setPageStep(1);
        aScrollBar[i]->setValue(pCameraCal->get_default_page(aScanType[i]));
    }

    //////////////////////////////////////////////////////////////////////////
    //focus in/out scene signal mapper
    QSignalMapper* pFocusInSignalMapper = new QSignalMapper();

    connect(m_pMPR00 , SIGNAL(focusInScene()) , pFocusInSignalMapper , SLOT(map()));
    pFocusInSignalMapper->setMapping(m_pMPR00 , QString(m_pMPR00->get_name().c_str()));
    connect(m_pMPR01 , SIGNAL(focusInScene()) , pFocusInSignalMapper , SLOT(map()));
    pFocusInSignalMapper->setMapping(m_pMPR01 , QString(m_pMPR01->get_name().c_str()));
    connect(m_pMPR10 , SIGNAL(focusInScene()) , pFocusInSignalMapper , SLOT(map()));
    pFocusInSignalMapper->setMapping(m_pMPR10 , QString(m_pMPR10->get_name().c_str()));

    connect(pFocusInSignalMapper , SIGNAL(mapped(QString)) , this , SLOT(slot_focus_in_scene_i(QString)));

    QSignalMapper* pFocusOutSignalMapper = new QSignalMapper();

    connect(m_pMPR00 , SIGNAL(focusOutScene()) , pFocusInSignalMapper , SLOT(map()));
    pFocusOutSignalMapper->setMapping(m_pMPR00 , QString(m_pMPR00->get_name().c_str()));
    connect(m_pMPR01 , SIGNAL(focusOutScene()) , pFocusInSignalMapper , SLOT(map()));
    pFocusOutSignalMapper->setMapping(m_pMPR01 , QString(m_pMPR01->get_name().c_str()));
    connect(m_pMPR10 , SIGNAL(focusOutScene()) , pFocusInSignalMapper , SLOT(map()));
    pFocusOutSignalMapper->setMapping(m_pMPR10 , QString(m_pMPR10->get_name().c_str()));

    connect(pFocusOutSignalMapper , SIGNAL(mapped(QString)) , this , SLOT(slot_focus_out_scene_i(QString)));
    //////////////////////////////////////////////////////////////////////////
}

void NoduleAnnotation::connectS_signal_slot_i()
{
    //Layout
    //connect(ui.action1x1 , SIGNAL(triggered()) , this , SLOT(SlotChangeLayout1x1_i()));
    connect(ui.action2x2 , SIGNAL(triggered()) , this , SLOT(slot_change_layout2x2_i()));

    //File
    connect(ui.actionOpen_DICOM_Folder , SIGNAL(triggered()) , this , SLOT(slot_open_dicom_folder_i()));
    connect(ui.actionOpen_Meta_Image , SIGNAL(triggered()) , this , SLOT(slot_open_meta_image_i()));
    connect(ui.actionOpen_Raw , SIGNAL(triggered()) , this , SLOT(slot_open_raw_i()));
    connect(ui.actionSave_Nodule , SIGNAL(triggered()) , this , SLOT(slot_save_nodule_i()));

    //MPR scroll bar
    connect(m_pMPR00ScrollBar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr00_i(int)));
    connect(m_pMPR01ScrollBar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr01_i(int)));
    connect(m_pMPR10ScrollBar , SIGNAL(valueChanged(int)) , this , SLOT(slot_sliding_bar_mpr10_i(int)));

    //Common tools
    connect(ui.pushButtonArrow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_arrow_i()));
    connect(ui.pushButtonAnnotate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_annotate_i()));
    connect(ui.pushButtonRotate , SIGNAL(pressed()) , this , SLOT(slot_press_btn_rotate_i()));
    connect(ui.pushButtonZoom , SIGNAL(pressed()) , this , SLOT(slot_press_btn_aoom_i()));
    connect(ui.pushButtonPan , SIGNAL(pressed()) , this , SLOT(slot_press_btn_pan_i()));
    connect(ui.pushButtonWindowing , SIGNAL(pressed()) , this , SLOT(slot_press_btn_windowing_i()));
    connect(ui.pushButtonFitWindow , SIGNAL(pressed()) , this , SLOT(slot_press_btn_fit_window_i()));

    //VOI list
    connect(ui.tableWidgetNoduleList , SIGNAL(cellPressed(int,int)) , this , SLOT(slot_voi_table_widget_cell_select_i(int ,int)));
    connect(ui.tableWidgetNoduleList , SIGNAL(itemChanged(QTableWidgetItem *)) , this , SLOT(slot_voi_table_widget_item_changed_i(QTableWidgetItem *)));
    connect(m_pNoduleObject , SIGNAL(nodule_added()) , this , SLOT(slot_add_nodule_i()));

    //Preset WL
    connect(ui.comboBoxPresetWL , SIGNAL(currentIndexChanged(QString)) , this , SLOT(slot_preset_wl_changed_i(QString)));

    //Scene Min Max hint
    connect(m_pMinMaxHintObject , SIGNAL(triggered(const std::string&)) , this , SLOT(slot_scene_min_max_hint_i(const std::string&)));

    //Crosshair visibility
    connect(ui.checkBoxCrossHair , SIGNAL(stateChanged(int)) , this , SLOT(slot_crosshair_visibility_i(int)));
}

void NoduleAnnotation::create_model_observer_i()
{
    //VOI
    m_pVOIModel.reset(new VOIModel());

    m_pVOITableOb.reset(new VOITableObserver());
    m_pVOITableOb->set_nodule_object(m_pNoduleObject);

    m_pSceneContainerOb.reset(new SceneContainerObserver());//万能 observer
    m_pSceneContainerOb->add_scene_container(m_pMPR00);
    m_pSceneContainerOb->add_scene_container(m_pMPR01);
    m_pSceneContainerOb->add_scene_container(m_pMPR10);

    m_pVOIModel->add_observer(m_pVOITableOb);
    //m_pVOIModel->add_observer(m_pSceneContainerOb);//Scene的刷新通过change item来完成

    //Crosshair & cross location
    m_pCrosshairModel.reset(new CrosshairModel());

    m_pMPRScrollBarOb.reset(new MPRScrollBarObserver());
    m_pMPRScrollBarOb->set_crosshair_model(m_pCrosshairModel);

    m_pCrosshairModel->add_observer(m_pMPRScrollBarOb);
    m_pCrosshairModel->add_observer(m_pSceneContainerOb);

    if (!m_pNoduleTypeSignalMapper)
    {
        delete m_pNoduleTypeSignalMapper;
        m_pNoduleTypeSignalMapper = new QSignalMapper(this);
        connect(m_pNoduleTypeSignalMapper , SIGNAL(mapped(int)) , this , SLOT(slot_voi_table_widget_nodule_type_changed_i(int)));
    }
}

void NoduleAnnotation::slot_change_layout2x2_i()
{
    if (!m_bReady)
    {
        return;
    }

    m_pMPR00->hide();
    m_pMPR00ScrollBar->hide();
    m_pMPR01->hide();
    m_pMPR01ScrollBar->hide();
    m_pMPR10->hide();
    m_pMPR10ScrollBar->hide();
    m_pVR11->hide();

    ui.gridLayout_6->removeWidget(m_pMPR00);
    ui.gridLayout_6->removeWidget(m_pMPR00ScrollBar);
    ui.gridLayout_6->removeWidget(m_pMPR01);
    ui.gridLayout_6->removeWidget(m_pMPR01ScrollBar);
    ui.gridLayout_6->removeWidget(m_pMPR10 );
    ui.gridLayout_6->removeWidget(m_pMPR10ScrollBar);
    ui.gridLayout_6->removeWidget(m_pVR11);

    ui.gridLayout_6->addWidget(m_pMPR00 , 0 ,0);
    ui.gridLayout_6->addWidget(m_pMPR00ScrollBar , 0 ,1,1,1);
    ui.gridLayout_6->addWidget(m_pMPR01 , 0 ,2);
    ui.gridLayout_6->addWidget(m_pMPR01ScrollBar , 0 ,3,1,1);
    ui.gridLayout_6->addWidget(m_pMPR10 , 1 ,0);
    ui.gridLayout_6->addWidget(m_pMPR10ScrollBar , 1 ,1,1,1);
    ui.gridLayout_6->addWidget(m_pVR11 , 1 ,2);

    m_pMPR00->show();
    m_pMPR00ScrollBar->show();
    m_pMPR01->show();
    m_pMPR01ScrollBar->show();
    m_pMPR10->show();
    m_pMPR10ScrollBar->show();
    m_pVR11->show();

    m_iLayoutType = 0;
}

void NoduleAnnotation::slot_open_dicom_folder_i()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    std::vector<QString> vecFileNames = fileNames.toVector().toStdVector();
    if (!vecFileNames.empty())
    {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        std::vector<std::string> vecSTDFiles;
        for (auto it = vecFileNames.begin() ; it != vecFileNames.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            std::cout << s << std::endl;
            vecSTDFiles.push_back(s);
        }

        std::shared_ptr<ImageDataHeader> pDataHeader;
        std::shared_ptr<ImageData> image_data;
        DICOMLoader loader;
        IOStatus status = loader.load_series(vecSTDFiles, image_data , pDataHeader);
        if (status != IO_SUCCESS)
        {
            QApplication::restoreOverrideCursor();
            QMessageBox::warning(this , tr("load DICOM Folder") , tr("load DICOM folder failed!"));
            return;
        }

        if (m_pVolumeInfos)//Delete last one
        {
            m_pVolumeInfos->finialize();
        }
        m_pVolumeInfos.reset(new VolumeInfos());
        m_pVolumeInfos->set_data_header(pDataHeader);
        //SharedWidget::instance()->makeCurrent();
        m_pVolumeInfos->set_volume(image_data);//load volume texture if has graphic card

        create_model_observer_i();

        create_scene_i();

        QApplication::restoreOverrideCursor();

        m_pMPR00->update();
        m_pMPR01->update();
        m_pMPR10->update();

        m_bReady = true;
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
    if (!m_bReady)
    {
        return;
    }

    std::vector<MPRScenePtr> vecMPRScenes;
    vecMPRScenes.push_back(m_pMPRScene00);
    vecMPRScenes.push_back(m_pMPRScene01);
    vecMPRScenes.push_back(m_pMPRScene10);

    std::vector<SceneContainer*> vecMPRs;
    vecMPRs.push_back(m_pMPR00);
    vecMPRs.push_back(m_pMPR01);
    vecMPRs.push_back(m_pMPR10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpAnnotate> pAnnotate(new MouseOpAnnotate());
        pAnnotate->set_scene(vecMPRScenes[i]);
        pAnnotate->set_voi_model(m_pVOIModel);//Set Model to annotate tools
        vecMPRs[i]->register_mouse_operation(pAnnotate , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_arrow_i()
{
    if (!m_bReady)
    {
        return;
    }

    std::vector<MPRScenePtr> vecMPRScenes;
    vecMPRScenes.push_back(m_pMPRScene00);
    vecMPRScenes.push_back(m_pMPRScene01);
    vecMPRScenes.push_back(m_pMPRScene10);

    std::vector<SceneContainer*> vecMPRs;
    vecMPRs.push_back(m_pMPR00);
    vecMPRs.push_back(m_pMPR01);
    vecMPRs.push_back(m_pMPR10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpLocate> pMPRLocate(new MouseOpLocate());
        pMPRLocate->set_scene(vecMPRScenes[i]);
        pMPRLocate->set_crosshair_model(m_pCrosshairModel);
        vecMPRs[i]->register_mouse_operation(pMPRLocate , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_rotate_i()
{
    if (!m_bReady)
    {
        return;
    }

    std::vector<MPRScenePtr> vecMPRScenes;
    vecMPRScenes.push_back(m_pMPRScene00);
    vecMPRScenes.push_back(m_pMPRScene01);
    vecMPRScenes.push_back(m_pMPRScene10);

    std::vector<SceneContainer*> vecMPRs;
    vecMPRs.push_back(m_pMPR00);
    vecMPRs.push_back(m_pMPR01);
    vecMPRs.push_back(m_pMPR10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpRotate> pRotate(new MouseOpRotate());
        pRotate->set_scene(vecMPRScenes[i]);
        vecMPRs[i]->register_mouse_operation(pRotate , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_aoom_i()
{
    if (!m_bReady)
    {
        return;
    }

    std::vector<MPRScenePtr> vecMPRScenes;
    vecMPRScenes.push_back(m_pMPRScene00);
    vecMPRScenes.push_back(m_pMPRScene01);
    vecMPRScenes.push_back(m_pMPRScene10);

    std::vector<SceneContainer*> vecMPRs;
    vecMPRs.push_back(m_pMPR00);
    vecMPRs.push_back(m_pMPR01);
    vecMPRs.push_back(m_pMPR10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpZoom> pZoom(new MouseOpZoom());
        pZoom->set_scene(vecMPRScenes[i]);
        vecMPRs[i]->register_mouse_operation(pZoom , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_pan_i()
{
    if (!m_bReady)
    {
        return;
    }

    std::vector<MPRScenePtr> vecMPRScenes;
    vecMPRScenes.push_back(m_pMPRScene00);
    vecMPRScenes.push_back(m_pMPRScene01);
    vecMPRScenes.push_back(m_pMPRScene10);

    std::vector<SceneContainer*> vecMPRs;
    vecMPRs.push_back(m_pMPR00);
    vecMPRs.push_back(m_pMPR01);
    vecMPRs.push_back(m_pMPR10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpPan> pPan(new MouseOpPan());
        pPan->set_scene(vecMPRScenes[i]);
        vecMPRs[i]->register_mouse_operation(pPan , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_windowing_i()
{
    if (!m_bReady)
    {
        return;
    }

    std::vector<MPRScenePtr> vecMPRScenes;
    vecMPRScenes.push_back(m_pMPRScene00);
    vecMPRScenes.push_back(m_pMPRScene01);
    vecMPRScenes.push_back(m_pMPRScene10);

    std::vector<SceneContainer*> vecMPRs;
    vecMPRs.push_back(m_pMPR00);
    vecMPRs.push_back(m_pMPR01);
    vecMPRs.push_back(m_pMPR10);

    for (int i = 0 ; i < 3 ; ++i)
    {
        std::shared_ptr<MouseOpWindowing> pWindowing(new MouseOpWindowing());
        pWindowing->set_scene(vecMPRScenes[i]);
        vecMPRs[i]->register_mouse_operation(pWindowing , Qt::LeftButton , Qt::NoModifier);
    }
}

void NoduleAnnotation::slot_press_btn_fit_window_i()
{
    //TODO
    if (!m_bReady)
    {
        return;
    }
}

void NoduleAnnotation::slot_save_nodule_i()
{
    if (!m_bReady)
    {
        return;
    }

    if (m_pVOIModel->get_voi_spheres().empty())
    {
        if(QMessageBox::No == QMessageBox::warning(
            this , tr("Save Nodule") , tr("Nodule count is zero. If you still want to save to file?"),QMessageBox::Yes |QMessageBox::No))
        {
            return;
        }
    }

    QString sFileCustom = QFileDialog::getSaveFileName(this, tr("Save Nodule") , QString(m_pVolumeInfos->get_data_header()->series_uid.c_str()), tr("NoduleSet(*.csv)"));
    if (!sFileCustom.isEmpty())
    {
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        const std::list<VOISphere>& voiList = m_pVOIModel->get_voi_spheres();
        for (auto it = voiList.begin() ; it != voiList.end() ; ++it)
        {
            nodule_set->add_nodule(*it);
        }

        NoduleSetCSVParser parser;
        std::string file_path(sFileCustom.toLocal8Bit());
        IOStatus status = parser.save(file_path , nodule_set);

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

void NoduleAnnotation::slot_sliding_bar_mpr00_i(int value)
{
    if(m_pCrosshairModel->page_to(m_pMPRScene00 , value))
    {
        m_pCrosshairModel->notify();
    }
}

void NoduleAnnotation::slot_sliding_bar_mpr01_i(int value)
{
    if(m_pCrosshairModel->page_to(m_pMPRScene01 , value))
    {
        m_pCrosshairModel->notify();
    }
}

void NoduleAnnotation::slot_sliding_bar_mpr10_i(int value)
{
    if(m_pCrosshairModel->page_to(m_pMPRScene10 , value))
    {
        m_pCrosshairModel->notify();
    }
}

void NoduleAnnotation::slot_voi_table_widget_cell_select_i(int row , int column)
{
    std::cout << "CellSelect "<< row << " " << column<< std::endl; 
    VOISphere voi = m_pVOIModel->get_voi_sphere(row);
    const Matrix4 matP2W = m_pMPRScene00->get_camera_calculator()->get_patient_to_world_matrix();
    m_pCrosshairModel->locate(matP2W.transform(voi.center));
    m_pCrosshairModel->notify();
}

void NoduleAnnotation::slot_voi_table_widget_item_changed_i(QTableWidgetItem *item)
{
    const int row = item->row();
    const int column = item->column();
    if (1 == column)
    {
        std::string sDiameter =  (item->text()).toLocal8Bit();
        StrNumConverter<double> con;
        m_pVOIModel->modify_voi_sphere_diameter(row , con.to_num(sDiameter));
        m_pSceneContainerOb->update();
    }
}

void NoduleAnnotation::slot_add_nodule_i()
{
    const std::list<VOISphere>& vecVOISphere = m_pVOIModel->get_voi_spheres();
    if (!vecVOISphere.empty())
    {
        ui.tableWidgetNoduleList->setRowCount(vecVOISphere.size());//Set row count , otherwise set item useless
        StrNumConverter<double> converter;
        const int iPrecision = 2;
        int iRow = 0;
        for (auto it = vecVOISphere.begin() ; it != vecVOISphere.end() ; ++it)
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

            QComboBox * pNoduleType = new QComboBox();
            pNoduleType->clear();
            pNoduleType->insertItem(0 ,  ksNoduleTypeGGN.c_str());
            pNoduleType->insertItem(1 , ksNoduleTypeAAH.c_str());
            //m_pTableWidgetVOI->setItem(iRow,2, new QTableWidgetItem("AAH"));
            ui.tableWidgetNoduleList->setCellWidget(iRow,2, pNoduleType);

            connect(pNoduleType , SIGNAL(currentIndexChanged(int)) , m_pNoduleTypeSignalMapper , SLOT(map()));
            m_pNoduleTypeSignalMapper->setMapping(pNoduleType , iRow);

            ++iRow;
        }
    }
}

void NoduleAnnotation::slot_delete_nodule_i(int id)
{

}

void NoduleAnnotation::slot_voi_table_widget_nodule_type_changed_i(int id)
{
    QWidget* pWidget = ui.tableWidgetNoduleList->cellWidget(id , 2);

    QComboBox* pBox= dynamic_cast<QComboBox*>(pWidget);
    if (pBox)
    {
        std::string type = pBox->currentText().toStdString();
        std::cout << id <<'\t' << type << std::endl;

        m_pVOIModel->modify_voi_sphere_name(id , type);
    }
}

void NoduleAnnotation::slot_preset_wl_changed_i(QString s)
{
    if (!m_bReady)
    {
        return;
    }

    std::string sWindow = std::string(s.toLocal8Bit());
    float fWW(1) , fWL(0);
    if (sWindow == std::string("CT_Lungs"))
    {
        fWW = kfPresetCTLungsWW;
        fWL   = kfPresetCTLungsWL;
    }
    else if (sWindow == std::string("CT_Chest"))
    {
        fWW = kfPresetCTChestWW;
        fWL   = kfPresetCTChestWL;
    }
    else if (sWindow == std::string("CT_Bone"))
    {
        fWW = kfPresetCTBoneWW;
        fWL   = kfPresetCTBoneWL;
    }
    else if (sWindow == std::string("CT_Angio"))
    {
        fWW = kfPresetCTAngioWW;
        fWL   = kfPresetCTAngioWL;
    }
    else if (sWindow == std::string("CT_Abdomen"))
    {
        fWW = kfPresetCTAbdomenWW;
        fWL   = kfPresetCTAbdomenWL;
    }
    else if (sWindow == std::string("CT_Brain"))
    {
        fWW = kfPresetCTBrainWW;
        fWL   = kfPresetCTBrainWL;
    }
    else 
    {
        return;
    }

    m_pMPRScene00->set_global_window_level(fWW , fWL);
    m_pMPRScene01->set_global_window_level(fWW , fWL);
    m_pMPRScene10->set_global_window_level(fWW , fWL);
    m_pSceneContainerOb->update();
}

void NoduleAnnotation::slot_scene_min_max_hint_i(const std::string& sName)
{
    if (!m_bReady)
    {
        return;
    }

    SceneContainer* pTragetContainer = nullptr;
    QScrollBar* pTargetBar = nullptr;
    if (0 == m_iLayoutType)
    {
        if (sName == m_pMPRScene00->get_name())
        {
            pTragetContainer = m_pMPR00;
            pTargetBar = m_pMPR00ScrollBar;
        }
        else if (sName == m_pMPRScene01->get_name())
        {
            pTragetContainer = m_pMPR01;
            pTargetBar = m_pMPR01ScrollBar;
        }
        else if (sName == m_pMPRScene10->get_name())
        {
            pTragetContainer = m_pMPR10;
            pTargetBar = m_pMPR10ScrollBar;
        }
        else
        {
            return;
        }

        m_pMPR00->hide();
        m_pMPR00ScrollBar->hide();
        m_pMPR01->hide();
        m_pMPR01ScrollBar->hide();
        m_pMPR10->hide();
        m_pMPR10ScrollBar->hide();
        m_pVR11->hide();

        ui.gridLayout_6->removeWidget(m_pMPR00);
        ui.gridLayout_6->removeWidget(m_pMPR00ScrollBar);
        ui.gridLayout_6->removeWidget(m_pMPR01);
        ui.gridLayout_6->removeWidget(m_pMPR01ScrollBar);
        ui.gridLayout_6->removeWidget(m_pMPR10 );
        ui.gridLayout_6->removeWidget(m_pMPR10ScrollBar);
        ui.gridLayout_6->removeWidget(m_pVR11);

        ui.gridLayout_6->addWidget(pTragetContainer , 0 ,0);
        ui.gridLayout_6->addWidget(pTargetBar , 0 ,1,1,1);

        pTragetContainer->show();
        pTargetBar->show();
        pTragetContainer->updateGL();

        m_iLayoutType = 1;

    }
    else
    {
        slot_change_layout2x2_i();
    }
}

void NoduleAnnotation::slot_focus_in_scene_i(QString s)
{
    if (!m_bReady)
    {
        return;
    }

    const std::string sName(s.toLocal8Bit());

    if (sName == m_pMPRScene00->get_name())
    {
        m_pCrosshairModel->focus(m_pMPRScene00);
    }
    else if (sName == m_pMPRScene01->get_name())
    {
        m_pCrosshairModel->focus(m_pMPRScene01);
    }
    else if (sName == m_pMPRScene10->get_name())
    {
        m_pCrosshairModel->focus(m_pMPRScene10);
    }
    else
    {

    }
}

void NoduleAnnotation::slot_focus_out_scene_i(QString sName)
{
    if (!m_bReady)
    {
        return;
    }

    m_pCrosshairModel->focus(nullptr);

}

void NoduleAnnotation::slot_crosshair_visibility_i(int iFlag)
{
    if (!m_bReady)
    {
        return;
    }

    m_pCrosshairModel->set_visibility(iFlag != 0);

    SceneContainer* aContainers[3] = {m_pMPR00 , m_pMPR01 , m_pMPR10};
    std::shared_ptr<MPRScene> aScenes[3] = {m_pMPRScene00 , m_pMPRScene01 , m_pMPRScene10};
    if (0 == iFlag)
    {
        for (int i = 0 ;i<3 ; ++i)
        {
            IMouseOpPtrCollection vecOps = aContainers[i]->get_mouse_operation(Qt::LeftButton ,  Qt::NoModifier);
            IMouseOpPtrCollection vecOpsNew;
            for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
            {
                if (typeid(*it).name() != typeid(MouseOpLocate*).name())
                {
                    vecOpsNew.push_back(*it);
                }
            }
            aContainers[i]->register_mouse_operation(vecOpsNew , Qt::LeftButton , Qt::NoModifier);
        }
    }
    else
    {
        for (int i = 0 ;i<3 ; ++i)
        {
            IMouseOpPtrCollection vecOps = aContainers[i]->get_mouse_operation(Qt::LeftButton ,  Qt::NoModifier);
            IMouseOpPtrCollection vecOpsNew;
            for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
            {
                if (typeid(*it).name() != typeid(std::shared_ptr<MouseOpLocate>).name())
                {
                    vecOpsNew.push_back(*it);
                }
            }
            std::shared_ptr<MouseOpLocate> pOpLocate(new MouseOpLocate());
            pOpLocate->set_scene(aScenes[i]);
            pOpLocate->set_crosshair_model(m_pCrosshairModel);
            vecOpsNew.push_back(pOpLocate);
            aContainers[i]->register_mouse_operation(vecOpsNew , Qt::LeftButton , Qt::NoModifier);
        }
    }
    m_pSceneContainerOb->update();
}





