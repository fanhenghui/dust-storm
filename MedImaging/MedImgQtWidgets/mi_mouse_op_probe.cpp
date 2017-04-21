#include "mi_mouse_op_probe.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgIO/mi_image_data.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpProbe::MouseOpProbe()
{

}

MouseOpProbe::~MouseOpProbe()
{

}

void MouseOpProbe::press(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    m_ptPre = pt;
}

void MouseOpProbe::move(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }

    //TODO MPR VR diverse strategy
    std::shared_ptr<MPRScene>  pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
    if (pScene)
    {
        Point3 ptV;
        if(pScene->get_volume_position(Point2(pt.x() , pt.y()) , ptV))
        {
            std::shared_ptr<VolumeInfos> pVolumeInfos = pScene->get_volume_infos();
            if (pVolumeInfos)
            {
                std::shared_ptr<ImageData> pImg = pVolumeInfos->get_volume();
                if (pImg)
                {
                    double dPixelValue(0);
                    pImg->get_pixel_value(ptV , dPixelValue);
                    dPixelValue =dPixelValue*pImg->_slope + pImg->_intercept;
                    std::cout <<dPixelValue << " " << ptV.x << " " << ptV.y << " " << ptV.z << std::endl;
                }
            }
            
        }
    }
    m_ptPre = pt;
}

void MouseOpProbe::release(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

void MouseOpProbe::double_click(const QPoint& pt)
{
    if (!m_pScene)
    {
        return;
    }
    m_ptPre = pt;
}

MED_IMAGING_END_NAMESPACE