#include "mi_observer_mpr_scroll_bar.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgQtWidgets/mi_model_cross_hair.h"

#include "qt/qscrollbar.h"

MPRScrollBarObserver::MPRScrollBarObserver()
{

}

MPRScrollBarObserver::~MPRScrollBarObserver()
{

}

void MPRScrollBarObserver::set_crosshair_model(std::shared_ptr<medical_imaging::CrosshairModel> model)
{
    _model = model;
}

void MPRScrollBarObserver::add_scroll_bar(std::shared_ptr<medical_imaging::MPRScene> scene, QScrollBar* widget)
{
    _scenes.push_back(scene);
    _scoll_bars.push_back(widget);
}

void MPRScrollBarObserver::update()
{
    //TODO check null
    std::shared_ptr<medical_imaging::CrosshairModel> model = _model.lock();
    if (model)
    {
        for (int i = 0; i<_scenes.size() ; ++i)
        {
            int page = model->get_page(_scenes[i]);
            if (page != _scoll_bars[i]->value())
            {
                _scoll_bars[i]->setValue(page);
            }
        }
    }
}
