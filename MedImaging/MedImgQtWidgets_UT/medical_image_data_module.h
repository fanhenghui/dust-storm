#ifndef MEDICAL_IMAGE_DATA_MODULE_H_
#define MEDICAL_IMAGE_DATA_MODULE_H_

#include "qt/qobject.h"

namespace medical_imaging
{
    class VolumeInfos;
    class MPRScene;
}

class SceneContainer;
class QAction;

class MyMainWindow;

class MedicalImageDataModule : public QObject
{
    Q_OBJECT
public:
    MedicalImageDataModule(
        MyMainWindow*main_window , 
        SceneContainer* scene  , 
        QAction* action_open_dicom , 
        QAction* action_open_meta,
        QObject* parent = 0);

protected slots:
    void slot_action_open_dicom_folder();
    void slot_action_open_meta_folder();
protected:

private:
    MyMainWindow* _main_window;
    SceneContainer* m_pMPRScene;
    std::shared_ptr<medical_imaging::VolumeInfos> m_pVolumeInfos;
};

#endif