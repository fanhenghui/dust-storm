#include "mi_mouse_op_annotate.h"

#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_vector3.h"

#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"

#include "mi_model_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

const std::string ksNoduleTypeGGN = std::string("GGN");
const std::string ksNoduleTypeAAH = std::string("AAH");

MouseOpAnnotate::MouseOpAnnotate():m_bPin(false),diameter(0.0)
{

}

MouseOpAnnotate::~MouseOpAnnotate()
{

}

void MouseOpAnnotate::press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_ptPre = pt;
    m_bPin = false;
    diameter = 0.0;

    //New voi
    std::shared_ptr<MPRScene>  pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
    if (pScene&&m_pVOIModel)
    {
        Point3 ptCenter;
        if(pScene->get_patient_position(Point2(pt.x() , pt.y()) , ptCenter))
        {
            //Get VOI center
            m_bPin = true;
            m_ptCenter = ptCenter;
            diameter = 0.0;
            m_pVOIModel->add_voi_sphere(medical_imaging::VOISphere(m_ptCenter , diameter , ksNoduleTypeGGN));
            m_pVOIModel->notify();
        }
    }
}

void MouseOpAnnotate::move(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    if (m_bPin&&m_pVOIModel)
    {
        Point3 ptFace;
        std::shared_ptr<MPRScene>  pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
        if(pScene->get_patient_position(Point2(pt.x() , pt.y()) , ptFace))
        {
            //Get VOI center
            Vector3 v = ptFace - m_ptCenter;
            diameter = v.magnitude()*2.0;

            m_pVOIModel->modify_voi_sphere_list_rear(medical_imaging::VOISphere(m_ptCenter , diameter));
            m_pVOIModel->notify();
        }
    }

    m_ptPre = pt;
}

void MouseOpAnnotate::release(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpAnnotate::double_click(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpAnnotate::set_voi_model(std::shared_ptr<VOIModel> pVOIModel)
{
    m_pVOIModel = pVOIModel;
}

MED_IMAGING_END_NAMESPACE