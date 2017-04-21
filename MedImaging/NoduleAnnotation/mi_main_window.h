#ifndef MI_MAIN_WINDOW_H
#define MI_MAIN_WINDOW_H

#include <QtGui/QMainWindow>
#include "ui_mi_main_window.h"

namespace medical_imaging
{
    class SceneBase;
    class MPRScene;
    class VolumeInfos;
    class VOIModel;
    class VOIObserver;
    class CrosshairModel;
    class SceneContainerObserver;
}
typedef std::shared_ptr<medical_imaging::MPRScene> MPRScenePtr;

class SceneContainer;

class VOITableObserver;
class MPRScrollBarObserver;

class QScrollBar;
class QSignalMapper;
class QNoduleObject;
class QMinMaxHintObject;

class NoduleAnnotation : public QMainWindow
{
    Q_OBJECT

public:
    NoduleAnnotation(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~NoduleAnnotation();

private slots:
    //Layout 
    void slot_change_layout2x2_i();

    //File
    void slot_open_dicom_folder_i();
    void slot_open_meta_image_i();
    void slot_open_raw_i();
    void slot_save_nodule_i();

    //Common tools
    void slot_press_btn_arrow_i();
    void slot_press_btn_annotate_i();
    void slot_press_btn_rotate_i();
    void slot_press_btn_aoom_i();
    void slot_press_btn_pan_i();
    void slot_press_btn_windowing_i();
    void slot_press_btn_fit_window_i();

    //MPR scroll bar 
    void slot_sliding_bar_mpr00_i(int value);
    void slot_sliding_bar_mpr01_i(int value);
    void slot_sliding_bar_mpr10_i(int value);

    //VOI list
    void slot_voi_table_widget_cell_select_i(int row , int column);
    void slot_voi_table_widget_item_changed_i(QTableWidgetItem *item);
    void slot_voi_table_widget_nodule_type_changed_i(int id);
    void slot_add_nodule_i();
    void slot_delete_nodule_i(int id);

    //Preset WL
    void slot_preset_wl_changed_i(QString s);

    //Min Max Hint
    void slot_scene_min_max_hint_i(const std::string& name);

    //focus In/Out Scene
    void slot_focus_in_scene_i(QString name);
    void slot_focus_out_scene_i(QString name);

    //Crosshair visible
    void slot_crosshair_visibility_i(int );


private:
    void connect_signal_slot_i();
    void configure_i();
    void create_scene_i();
    void create_model_observer_i();

private:
    Ui::NoduleAnnotationClass ui;

private:
    bool m_bReady;

    SceneContainer* m_pMPR00;
    SceneContainer* m_pMPR01;
    SceneContainer* m_pMPR10;
    SceneContainer* m_pVR11;

    QScrollBar * m_pMPR00ScrollBar;
    QScrollBar * m_pMPR01ScrollBar;
    QScrollBar * m_pMPR10ScrollBar;

    std::shared_ptr<medical_imaging::VolumeInfos> m_pVolumeInfos;
    MPRScenePtr m_pMPRScene00;
    MPRScenePtr m_pMPRScene01;
    MPRScenePtr m_pMPRScene10;

    //Layout Type
    //0 2x2
    //1 1x1
    int m_iLayoutType;

    //Model
    std::shared_ptr<medical_imaging::VOIModel> m_pVOIModel;
    std::shared_ptr<medical_imaging::CrosshairModel> m_pCrosshairModel;

    //Observer
    std::shared_ptr<VOITableObserver> m_pVOITableOb;
    std::shared_ptr<medical_imaging::SceneContainerObserver> m_pSceneContainerOb;
    std::shared_ptr<MPRScrollBarObserver> m_pMPRScrollBarOb;

    //Nodule VOI list
    QSignalMapper* m_pNoduleTypeSignalMapper;
    QNoduleObject* m_pNoduleObject;

    //Scene min max hint
    QMinMaxHintObject* m_pMinMaxHintObject;
};

#endif // MI_MAIN_WINDOW_H
