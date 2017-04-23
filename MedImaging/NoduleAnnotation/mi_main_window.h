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
    class ProgressModel;
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
    void slot_save_nodule_file_i();
    void slot_open_nodule_file_i();

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
    bool _is_ready;

    SceneContainer* _mpr_00;
    SceneContainer* _mpr_01;
    SceneContainer* _mpr10;
    SceneContainer* _vr_11;

    QScrollBar * _mpr_00_scroll_bar;
    QScrollBar * _mpr_01_scroll_bar;
    QScrollBar * _mpr_10_scroll_bar;

    std::shared_ptr<medical_imaging::VolumeInfos> _volume_infos;
    MPRScenePtr _mpr_scene_00;
    MPRScenePtr _mpr_scene_01;
    MPRScenePtr _mpr_scene_10;

    //Layout Type
    //0 2x2
    //1 1x1
    int _layout_tag;

    //Model
    std::shared_ptr<medical_imaging::VOIModel> _model_voi;
    std::shared_ptr<medical_imaging::CrosshairModel> _model_crosshari;
    std::shared_ptr<medical_imaging::ProgressModel> _model_progress;

    //Observer
    std::shared_ptr<VOITableObserver> _ob_voi_table;
    std::shared_ptr<medical_imaging::SceneContainerObserver> _ob_scene_container;
    std::shared_ptr<MPRScrollBarObserver> _ob_mpr_scroll_bar;

    //Nodule VOI list
    QSignalMapper* _single_manager_nodule_type;
    QNoduleObject* _object_nodule;

    //Scene min max hint
    QMinMaxHintObject* _object_min_max_hint;

};

#endif // MI_MAIN_WINDOW_H
