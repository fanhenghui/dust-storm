#include "mi_mouse_op_annotate_fine_tuning.h"

#include "renderalgo/mi_scene_base.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "arithmetic/mi_aabb.h"

#include "mi_model_voi.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpAnnotateFineTuning::MouseOpAnnotateFineTuning() : 
        _tune_type(MouseOpAnnotateFineTuning::SUBSTRACT), 
        _shape_type(MouseOpAnnotateFineTuning::CUBE),
        _tune_status(0)
{

}
MouseOpAnnotateFineTuning::~MouseOpAnnotateFineTuning()
{
    ;
}
void MouseOpAnnotateFineTuning::press(const QPointF& pt)
{
    // display the default circle at specified location
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    this->_tune_status = 1;
    this->tune_voi(pt);

}

void MouseOpAnnotateFineTuning::move(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    std::shared_ptr<MPRScene>  slice = std::dynamic_pointer_cast<MPRScene>(scene_base);
    // draw the circle to represent
    Point3 physical_pos;
    if (slice->get_world_position(Point2(pt.x() , pt.y()) , physical_pos))
    {
        this->_model->set_tune_location(physical_pos);
    }

    if (this->_tune_status)
    {
        this->tune_voi(pt);
    }
}

void MouseOpAnnotateFineTuning::tune_voi(const QPointF &pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    std::shared_ptr<MPRScene> slice = std::dynamic_pointer_cast<MPRScene>(scene_base);
    if (slice && this->_model)
    {
        Point3 voxel_idx;
        if(slice->get_volume_position(Point2(pt.x() , pt.y()) , voxel_idx))
        {
            //double r = this->_model->get_tune_radius(); // TODO: user can change it
            //unsigned int begin[3] = {
            //    std::max(voxel_idx.x-r,0.0), std::max(voxel_idx.y-r,0.0), std::max(voxel_idx.z-r,0.0) };
            //    unsigned int end[3] = {voxel_idx.x+r, voxel_idx.y+r, voxel_idx.z+r};
            //    // TODO: this erase_block is in !!!volume!!! space, which actually should be in image space
            //    AABBUI erase_block(begin, end);
            //    this->_model->set_voxel_block_to_tune(erase_block, MouseOpAnnotateFineTuning::SUBSTRACT); // -1:minus +1:add
            
            std::vector<unsigned int> idx; idx.reserve(3);
            idx.push_back(voxel_idx.x);idx.push_back(voxel_idx.y);idx.push_back(voxel_idx.z);
            this->_model->set_voxel_to_tune(idx, MouseOpAnnotateFineTuning::SUBSTRACT);
            this->_model->notify(VOIModel::TUNING_VOI);
        }
    }
}

void MouseOpAnnotateFineTuning::release(const QPointF& pt)
{
    this->_tune_status = 0;
}

void MouseOpAnnotateFineTuning::double_click(const QPointF& pt)
{

}

void MouseOpAnnotateFineTuning::wheel_slide(int /*value*/)
{

}

void MouseOpAnnotateFineTuning::set_voi_model(std::shared_ptr<VOIModel> model)
{
    this->_model = std::move(model);
}

MED_IMG_END_NAMESPACE