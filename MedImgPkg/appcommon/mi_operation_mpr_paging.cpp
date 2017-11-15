#include "mi_operation_mpr_paging.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "io/mi_image_data.h"

#include "arithmetic/mi_ortho_camera.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_message.pb.h"
#include "mi_app_common_logger.h"
#include "mi_model_crosshair.h"
#include "mi_app_common_define.h"

MED_IMG_BEGIN_NAMESPACE

OpMPRPaging::OpMPRPaging() {

}

OpMPRPaging::~OpMPRPaging() {
}

int OpMPRPaging::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN OpMPRPaging";
    //Parse data
    const unsigned int cell_id = _header.cell_id;
    int page_step = 1;

    if (_buffer != nullptr) {
        MsgMouse msg;
        if (!msg.ParseFromArray(_buffer , _header.data_len)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "parse mouse message failed in mpr paging.";
            return -1;
        }
        page_step = static_cast<int>(msg.cur().y() - msg.pre().y());
        msg.Clear();
        //MI_APPCOMMON_LOG(M/I_DEBUG) << "paging step : " << page_step;
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

    std::shared_ptr<IModel> model_crosshair_i = controller->get_model(MODEL_ID_CROSSHAIR);
    if (!model_crosshair_i) {
        MI_APPCOMMON_LOG(MI_ERROR) << "crosshair model null.";
        return -1;
    }
    std::shared_ptr<ModelCrosshair> model_crosshair = std::dynamic_pointer_cast<ModelCrosshair>(model_crosshair_i);
    if (!model_crosshair) {
        MI_APPCOMMON_LOG(MI_ERROR) << "error model id to acquire crosshair model.";
        return -1;
    }
    
    model_crosshair->page(mpr_scene, page_step);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT OpMPRPaging";
    return 0;
}


MED_IMG_END_NAMESPACE