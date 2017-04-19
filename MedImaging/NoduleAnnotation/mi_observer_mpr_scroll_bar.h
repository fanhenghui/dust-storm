#ifndef MED_IMAGING_VOBSERVER_MPR_SCROLL_BAR_H_
#define MED_IMAGING_VOBSERVER_MPR_SCROLL_BAR_H_

#include <memory>
#include <vector>
#include "MedImgCommon/mi_observer_interface.h"

namespace MedImaging
{
    class MPRScene;
    class CrosshairModel;
}

class QScrollBar;
class MPRScrollBarObserver : public MedImaging::IObserver
{
public:
    MPRScrollBarObserver();
    virtual ~MPRScrollBarObserver();
    void SetCrosshairModel(std::shared_ptr<MedImaging::CrosshairModel> pModel);
    void AddScrollBar(std::shared_ptr<MedImaging::MPRScene> pScene, QScrollBar* pWidget);
    virtual void Update();
protected:
private:
    std::weak_ptr<MedImaging::CrosshairModel> m_pModel;
    std::vector<std::shared_ptr<MedImaging::MPRScene>> m_vecScenes;
    std::vector<QScrollBar*> m_vecScrollBars;
};

#endif