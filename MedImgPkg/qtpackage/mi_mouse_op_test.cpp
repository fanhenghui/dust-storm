#include "mi_mouse_op_test.h"
#include "renderalgo/mi_scene_base.h"
#include "arithmetic/mi_point2.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "io/mi_image_data.h"
#include "mi_qt_package_logger.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpTest::MouseOpTest()
{

}

MouseOpTest::~MouseOpTest()
{

}

void MouseOpTest::press(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
    if (scene)
    {
        Point3 pt_v;
        if(scene->get_volume_position(Point2(pt.x() , pt.y()) , pt_v))
        {
            std::shared_ptr<VolumeInfos> volume_infos = scene->get_volume_infos();
            if (volume_infos)
            {
                std::shared_ptr<ImageData> volume_img = volume_infos->get_volume();
                std::shared_ptr<ImageData> mask_img = volume_infos->get_mask();
                double pixel_value(0);
                mask_img->get_pixel_value(pt_v ,pixel_value);
                MI_QTPACKAGE_LOG(MI_DEBUG) << "Mask Info : ( " << pt_v.x << " , " << pt_v.y << " , " << pt_v.z << " )  " << pixel_value;
                Point3 entry_point = scene->get_entry_point(pt.x() , pt.y());
                double pixel_value1(0);
                mask_img->get_pixel_value(entry_point ,pixel_value1);
                MI_QTPACKAGE_LOG(MI_DEBUG) << "Entry point : ( " << entry_point.x << " , " << entry_point.y << " , " << entry_point.z << " ) " << pixel_value1;
            }

        }
    }

    _pre_point = pt;
}

void MouseOpTest::move(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpTest::release(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpTest::double_click(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpTest::wheel_slide(int)
{

}

MED_IMG_END_NAMESPACE