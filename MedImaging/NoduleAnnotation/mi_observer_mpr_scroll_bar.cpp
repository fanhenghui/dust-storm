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

void MPRScrollBarObserver::SetCrosshairModel(std::shared_ptr<MedImaging::CrosshairModel> pModel)
{
    m_pModel = pModel;
}

void MPRScrollBarObserver::AddScrollBar(std::shared_ptr<MedImaging::MPRScene> pScene, QScrollBar* pWidget)
{
    m_vecScenes.push_back(pScene);
    m_vecScrollBars.push_back(pWidget);
}

void MPRScrollBarObserver::Update()
{
    //TODO check null
    std::shared_ptr<MedImaging::CrosshairModel> pModel = m_pModel.lock();
    if (pModel)
    {
        for (int i = 0; i<m_vecScenes.size() ; ++i)
        {
            int iPage = pModel->GetPage(m_vecScenes[i]);
            if (iPage != m_vecScrollBars[i]->value())
            {
                m_vecScrollBars[i]->setValue(iPage);
            }
        }
    }
}
