#ifndef MI_MAIN_WINDOW_H
#define MI_MAIN_WINDOW_H

#include <QtGui/QMainWindow>
#include "ui_mi_main_window.h"

namespace MedImaging
{
    class SceneBase;
    class MPRScene;
    class VolumeInfos;
    class VOIModel;
    class VOIObserver;
    class CrosshairModel;
    class SceneContainerObserver;
}
typedef std::shared_ptr<MedImaging::MPRScene> MPRScenePtr;

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

protected slots:
    //Layout 
    void SlotChangeLayout2x2_i();

    //File
    void SlotOpenDICOMFolder_i();
    void SlotOpenMetaImage_i();
    void SlotOpenRaw_i();
    void SlotSaveNodule_i();

    //Common tools
    void SlotPressBtnArrow_i();
    void SlotPressBtnAnnotate_i();
    void SlotPressBtnRotate_i();
    void SlotPressBtnZoom_i();
    void SlotPressBtnPan_i();
    void SlotPressBtnWindowing_i();
    void SlotPressBtnFitWindow_i();

    //MPR scroll bar 
    void SlotSlidingBarMPR00_i(int value);
    void SlotSlidingBarMPR01_i(int value);
    void SlotSlidingBarMPR10_i(int value);

    //VOI list
    void SlotVOITableWidgetCellSelect_i(int row , int column);
    void SlotVOITableWidgetItemChanged_i(QTableWidgetItem *item);
    void SlotVOITableWidgetNoduleTypeChanged_i(int id);
    void SlotVOIAddNodule_i();
    void SlotVOIDeleteNodule_i(int id);

    //Preset WL
    void SlotPersetWLChanged_i(QString s);

    //Min Max Hint
    void SlotSceneMinMaxHint_i(const std::string& sName);

    //Focus In/Out Scene
    void SlotFocusInScene_i(QString sName);
    void SlotFocusOutScene_i(QString sName);

    //Crosshair visible
    void SlotCrosshairVisbility_i(int );


private:
    void ConnectSignalSlot_i();
    void Configure_i();
    void CreateScene_i();
    void CreateModelObserver_i();

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

    std::shared_ptr<MedImaging::VolumeInfos> m_pVolumeInfos;
    MPRScenePtr m_pMPRScene00;
    MPRScenePtr m_pMPRScene01;
    MPRScenePtr m_pMPRScene10;

    //Layout Type
    //0 2x2
    //1 1x1
    int m_iLayoutType;

    //Model
    std::shared_ptr<MedImaging::VOIModel> m_pVOIModel;
    std::shared_ptr<MedImaging::CrosshairModel> m_pCrosshairModel;

    //Observer
    std::shared_ptr<VOITableObserver> m_pVOITableOb;
    std::shared_ptr<MedImaging::SceneContainerObserver> m_pSceneContainerOb;
    std::shared_ptr<MPRScrollBarObserver> m_pMPRScrollBarOb;

    //Nodule VOI list
    QSignalMapper* m_pNoduleTypeSignalMapper;
    QNoduleObject* m_pNoduleObject;

    //Scene min max hint
    QMinMaxHintObject* m_pMinMaxHintObject;
};

#endif // MI_MAIN_WINDOW_H
