#include "mi_be_operation_fe_mpr_mask_overlay.h"

#include "io/mi_message.pb.h"

#include "renderalgo/mi_mpr_scene.h"

#include "mi_app_cell.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFEMPRMaskOverlay::BEOpFEMPRMaskOverlay() {

}

BEOpFEMPRMaskOverlay::~BEOpFEMPRMaskOverlay() {

}

int BEOpFEMPRMaskOverlay::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFEMPRMaskOverlay.";
    if (_buffer == nullptr || _header.data_len < 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "incompleted mask overlay message.";
        return -1;
    }

    MsgMPRMaskOverlay msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse MPR mask overlay message failed.";
        return -1;
    }
    const int flag = msg.flag();
    const float opacity = msg.opacity();
    msg.Clear();
    
    const MaskOverlayMode mask_overlay_mode = flag != 0 ? MASK_OVERLAY_ENABLE : MASK_OVERLAY_DISABLE;

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::map<unsigned int, std::shared_ptr<AppCell>> cells = controller->get_cells();
    for (auto it = cells.begin(); it != cells.end(); ++it) {
        std::shared_ptr<SceneBase> scene = it->second->get_scene();
        std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
        if (mpr_scene) {
            mpr_scene->set_mask_overlay_mode(mask_overlay_mode);
            mpr_scene->set_mask_overlay_opacity(opacity);
            mpr_scene->set_dirty(true);
        } 
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFEMPRMaskOverlay.";
    return 0;
}

MED_IMG_END_NAMESPACE