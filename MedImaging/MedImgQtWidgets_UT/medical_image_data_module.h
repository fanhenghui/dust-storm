#ifndef MEDICAL_IMAGE_DATA_MODULE_H_
#define MEDICAL_IMAGE_DATA_MODULE_H_

#include "qt/qobject.h"

namespace MedImaging
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
        MyMainWindow*pMainWindow , 
        SceneContainer* pScene  , 
        QAction* pOpenDICOMFolder , 
        QAction* pOpenMetaFolder,
        QObject* parent = 0);

protected slots:
    void SlotActionOpenDICOMFolder();
    void SlotActionOpenMetaFolder();
protected:

private:
    MyMainWindow* m_pMainWindow;
    SceneContainer* m_pMPRScene;
    std::shared_ptr<MedImaging::VolumeInfos> m_pVolumeInfos;
};

#endif