#include "mi_operation_mpr_paging.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "io/mi_image_data.h"

#include "arithmetic/mi_ortho_camera.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

OpMPRPaging::OpMPRPaging() {

}

OpMPRPaging::~OpMPRPaging() {

}

int OpMPRPaging::execute() {
    //Parse data
    const unsigned int cell_id = _header._cell_id;
    int page_num = 1;

    if (_buffer != nullptr) {
        MsgPaging msg;
        if (msg.ParseFromArray(_buffer , _header._data_len)) {
            page_num = msg.page();
            std::cout << "paging num : " << page_num << std::endl;
        } else {
            MsgMouse msg;
            if (msg.ParseFromArray(_buffer , _header._data_len)) {
                page_num = static_cast<int>(msg.cur().y() - msg.pre().y());
                std::cout << "paging num : " << page_num << std::endl;
            }
        }
    }

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    APPCOMMON_CHECK_NULL_EXCEPTION(cell);

    std::shared_ptr<SceneBase> scene = cell->get_scene();
    APPCOMMON_CHECK_NULL_EXCEPTION(scene);

    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
    APPCOMMON_CHECK_NULL_EXCEPTION(mpr_scene);

    std::shared_ptr<VolumeInfos> volumeinfos = controller->get_volume_infos();
    std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>
                                          (mpr_scene->get_camera());
    int cur_page = mpr_scene->get_camera_calculator()->get_orthognal_mpr_page(camera);
    mpr_scene->page(page_num);
    return 0;
}


MED_IMG_END_NAMESPACE