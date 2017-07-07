#include "mi_operation_mpr_paging.h"

#include "MedImgAppCommon/mi_app_controller.h"
#include "MedImgAppCommon/mi_app_cell.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

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

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    REVIEW_CHECK_NULL_EXCEPTION(cell);

    std::shared_ptr<SceneBase> scene = cell->get_scene();
    REVIEW_CHECK_NULL_EXCEPTION(scene);

    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
    REVIEW_CHECK_NULL_EXCEPTION(mpr_scene);

    mpr_scene->page(1);
    mpr_scene->set_dirty(true);

    std::cout << "MPR paging done\n";

    return 0;
}


MED_IMG_END_NAMESPACE