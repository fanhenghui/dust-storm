#include "mi_be_operation_fe_windowing.h"

#include "arithmetic/mi_ortho_camera.h"

#include "io/mi_image_data.h"
#include "io/mi_protobuf.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFEWindowing::BEOpFEWindowing() {}

BEOpFEWindowing::~BEOpFEWindowing() {}

int BEOpFEWindowing::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFEWindowing.";
    const unsigned int cell_id = _header.cell_id;
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgMouse msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse windowing mouse message failed.";
        return -1;
    }
    const float pre_x = msg.pre().x();
    const float pre_y = msg.pre().y();
    const float cur_x = msg.cur().x();
    const float cur_y = msg.cur().y();
    msg.Clear();
    
    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    APPCOMMON_CHECK_NULL_EXCEPTION(cell);

    std::shared_ptr<SceneBase> scene = cell->get_scene();
    APPCOMMON_CHECK_NULL_EXCEPTION(scene);

    std::shared_ptr<VRScene> scene_ext =
        std::dynamic_pointer_cast<VRScene>(scene);

    if (scene_ext) {
        float ww, wl;
        if (scene_ext->get_composite_mode() == COMPOSITE_DVR) {
            // TODO consider when has mask
            scene_ext->get_window_level(ww, wl, 0);
            ww += (cur_x - pre_x);
            wl += (pre_y - cur_y);
            scene_ext->set_window_level(ww, wl, 0);

            //TODO harding coding
            if(scene_ext->get_mask_mode() != MASK_NONE) {
                scene_ext->get_window_level(ww, wl, 1);
                ww += (cur_x - pre_x);
                wl += (pre_y - cur_y);
                scene_ext->set_window_level(ww, wl, 1);
            }
        } else {
            scene_ext->get_global_window_level(ww, wl);
            ww += (cur_x - pre_x);
            wl += (pre_y - cur_y);
            scene_ext->set_global_window_level(ww, wl);
        }
    } else {
        std::shared_ptr<MPRScene> scene_ext =
            std::dynamic_pointer_cast<MPRScene>(scene);
        float ww, wl;
        scene_ext->get_global_window_level(ww, wl);
        ww += (cur_x - pre_x);
        wl += (pre_y - cur_y);
        scene_ext->set_global_window_level(ww, wl);
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFEWindowing.";
    return 0;
}

MED_IMG_END_NAMESPACE