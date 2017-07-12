#include "mi_operation_mpr_paging.h"

#include "MedImgAppCommon/mi_app_controller.h"
#include "MedImgAppCommon/mi_app_cell.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgIO/mi_image_data.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "mi_review_controller.h"

MED_IMG_BEGIN_NAMESPACE

OpMPRPaging::OpMPRPaging()
{

}

OpMPRPaging::~OpMPRPaging()
{

}

int OpMPRPaging::execute()
{
    //Parse data
    const unsigned int cell_id = _header._cell_id;

    std::shared_ptr<AppController> controller = _controller.lock();
    REVIEW_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ReviewController> review_controller = std::dynamic_pointer_cast<ReviewController>(controller);
    REVIEW_CHECK_NULL_EXCEPTION(review_controller);

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    REVIEW_CHECK_NULL_EXCEPTION(cell);

    std::shared_ptr<SceneBase> scene = cell->get_scene();
    REVIEW_CHECK_NULL_EXCEPTION(scene);

    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
    REVIEW_CHECK_NULL_EXCEPTION(mpr_scene);

    // mpr_scene->page(1);
    // mpr_scene->set_dirty(true);

    //////////////////////////////////////////////////
    //For testing
    std::shared_ptr<VolumeInfos> volumeinfos = review_controller->get_volume_infos();
    std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(mpr_scene->get_camera());
    int cur_page = mpr_scene->get_camera_calculator()->get_orthognal_mpr_page(camera);
    std::cout << "current page : " << cur_page << std::endl;
    if(cur_page >= volumeinfos->get_volume()->_dim[2] - 2)
    {
        mpr_scene->page_to(1);
    }
    else
    {
        mpr_scene->page(1);
    }

    return 0;
}


MED_IMG_END_NAMESPACE