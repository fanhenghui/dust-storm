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

void MPRScrollBarObserver::set_crosshair_model(std::shared_ptr<medical_imaging::CrosshairModel> pModel)
{
    m_pModel = pModel;
}

void MPRScrollBarObserver::add_scroll_bar(std::shared_ptr<medical_imaging::MPRScene> pScene, QScrollBar* pWidget)
{
    m_vecScenes.push_back(pScene);
    m_vecScrollBars.push_back(pWidget);
}

void MPRScrollBarObserver::update()
{
    //TODO check null
    std::shared_ptr<medical_imaging::CrosshairModel> pModel = m_pModel.lock();
    if (pModel)
    {
        for (int i = 0; i<m_vecScenes.size() ; ++i)
        {
            int iPage = pModel->get_page(m_vecScenes[i]);
            if (iPage != m_vecScrollBars[i]->value())
            {
                m_vecScrollBars[i]->setValue(iPage);
            }
        }
    }
}
