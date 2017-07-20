#include "mi_mouse_op_annotate_fine_tuning.h"

#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgArithmetic/mi_aabb.h"

#include "mi_model_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

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
    // std::cout << "click voxel at x: " << pt.x() << ", y: " << pt.y() << std::endl;
    if (!this->_scene)
    {
        return;
    }

    this->_tune_status = 1;
}

void MouseOpAnnotateFineTuning::move(const QPointF& pt)
{
    if (!this->_scene)
    {
        return;
    }
    if (this->_tune_status)
    {
        std::shared_ptr<MPRScene>  slice = std::dynamic_pointer_cast<MPRScene>(_scene);

        if (slice && this->_model)
        {
            Point3 voxel_idx;
            if(slice->get_volume_position(Point2(pt.x() , pt.y()) , voxel_idx))
            {
                double r = this->_model->get_tune_radius(); // TODO: user can change it
                unsigned int begin[3] = {
                    std::max(voxel_idx.x-r,0.0), std::max(voxel_idx.y-r,0.0), std::max(voxel_idx.z-r,0.0) };
                    unsigned int end[3] = {voxel_idx.x+r, voxel_idx.y+r, voxel_idx.z+r};
                    AABBUI erase_block(begin, end);
                    this->_model->set_voxel_to_tune(erase_block, MouseOpAnnotateFineTuning::SUBSTRACT); // -1:minus +1:add
                    this->_model->notify(VOIModel::TUNING_VOI);
            }
        }
    }
    else
    {
        //TODO: draw the cube to represent 
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

MED_IMAGING_END_NAMESPACE