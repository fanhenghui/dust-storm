#include "mi_operation_mpr_paging.h"

#include "MedImgAppCommon/mi_app_controller.h"
#include "MedImgAppCommon/mi_app_cell.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "io/mi_image_data.h"

#include "arithmetic/mi_ortho_camera.h"

#include "mi_review_controller.h"
#include "mi_message.pb.h"

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
    int page_num = 1;
    if(_buffer != nullptr){
        MsgPaging msg;
        if(msg.ParseFromArray(_buffer , _header._data_len)){
            page_num = msg.page();
            std::cout << "paging num : " << page_num << std::endl;
        }
    }

    std::shared_ptr<AppController> controller = _controller.lock();
    REVIEW_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ReviewController> review_controller = std::dynamic_pointer_cast<ReviewController>(controller);
    REVIEW_CHECK_NULL_EXCEPTION(review_controller);

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    REVIEW_CHECK_NULL_EXCEPTION(cell);

    std::shared_ptr<SceneBase> scene = cell->get_scene();
    REVIEW_CHECK_NULL_EXCEPTION(scene);

#ifdef MPR

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
        mpr_scene->page(page_num);
    }
#else
    std::shared_ptr<VRScene> vr_scene = std::dynamic_pointer_cast<VRScene>(scene);
    REVIEW_CHECK_NULL_EXCEPTION(vr_scene);

    std::shared_ptr<VolumeInfos> volumeinfos = review_controller->get_volume_infos();
    std::shared_ptr<CameraBase> camera = vr_scene->get_camera();
    
    Quat4 q(5.0/360.0*2.0*3.1415926 , Vector3(0,1,0));
    camera->rotate(q);
    vr_scene->set_dirty(true);
    
#endif
    return 0;
}


MED_IMG_END_NAMESPACE