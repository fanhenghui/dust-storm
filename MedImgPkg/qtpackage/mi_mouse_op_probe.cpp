#include "mi_mouse_op_probe.h"
#include "renderalgo/mi_scene_base.h"
#include "arithmetic/mi_point2.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "io/mi_image_data.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpProbe::MouseOpProbe()
{

}

MouseOpProbe::~MouseOpProbe()
{

}

void MouseOpProbe::press(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    _pre_point = pt;
}

void MouseOpProbe::move(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    //TODO MPR VR diverse strategy
    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
    if (scene)
    {
        Point3 pt_v;
        if(scene->get_volume_position(Point2(pt.x() , pt.y()) , pt_v))
        {
            std::shared_ptr<VolumeInfos> volume_infos = scene->get_volume_infos();
            if (volume_infos)
            {
                std::shared_ptr<ImageData> pImg = volume_infos->get_volume();
                if (pImg)
                {
                    double pixel_value(0);
                    pImg->get_pixel_value(pt_v , pixel_value);
                    pixel_value =pixel_value*pImg->_slope + pImg->_intercept;
                    //MI_QTPACKAGE_LOG(MI_DEBUG) <<pixel_value << " " << pt_v.x << " " << pt_v.y << " " << pt_v.z;
                }
            }
            
        }
    }
    _pre_point = pt;
}

void MouseOpProbe::release(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpProbe::double_click(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpProbe::wheel_slide(int)
{

}

MED_IMG_END_NAMESPACE