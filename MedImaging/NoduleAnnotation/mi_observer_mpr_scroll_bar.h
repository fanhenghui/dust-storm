#ifndef MED_IMAGING_VOBSERVER_MPR_SCROLL_BAR_H_
#define MED_IMAGING_VOBSERVER_MPR_SCROLL_BAR_H_

#include <memory>
#include <vector>
#include "MedImgCommon/mi_observer_interface.h"

namespace medical_imaging
{
    class MPRScene;
    class CrosshairModel;
}

class QScrollBar;
class MPRScrollBarObserver : public medical_imaging::IObserver
{
public:
    MPRScrollBarObserver();
    virtual ~MPRScrollBarObserver();
    void set_crosshair_model(std::shared_ptr<medical_imaging::CrosshairModel> pModel);
    void add_scroll_bar(std::shared_ptr<medical_imaging::MPRScene> pScene, QScrollBar* pWidget);
    virtual void update();
protected:
private:
    std::weak_ptr<medical_imaging::CrosshairModel> m_pModel;
    std::vector<std::shared_ptr<medical_imaging::MPRScene>> m_vecScenes;
    std::vector<QScrollBar*> m_vecScrollBars;
};

#endif