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

    void set_crosshair_model(std::shared_ptr<medical_imaging::CrosshairModel> model);

    void add_scroll_bar(std::shared_ptr<medical_imaging::MPRScene> scene, QScrollBar* widget);

    virtual void update(int code_id = 0);
protected:
private:
    std::weak_ptr<medical_imaging::CrosshairModel> _model;
    std::vector<std::shared_ptr<medical_imaging::MPRScene>> _scenes;
    std::vector<QScrollBar*> _scoll_bars;
};

#endif