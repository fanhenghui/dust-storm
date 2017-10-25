#include "mi_operation_switch_preset_vrt.h"

#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_opacity_transfer_function.h"
#include "renderalgo/mi_transfer_function_loader.h"
#include "renderalgo/mi_vr_scene.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_message.pb.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

OpSwitchPresetVRT::OpSwitchPresetVRT() {}

OpSwitchPresetVRT::~OpSwitchPresetVRT() {}

int OpSwitchPresetVRT::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN OpSwitchPresetVRT.";
    const unsigned int cell_id = _header._cell_id;
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;
    if (!msg.ParseFromArray(_buffer, _header._data_len)) {
        APPCOMMON_THROW_EXCEPTION("parse switch preset windowing message failed!");
    }

    const std::string context = msg.context();
    msg.Clear();
    
    static const int S_PRESET_VRT_NUM = 2;
    static const std::string S_PRESET_VRT_NAME[S_PRESET_VRT_NUM]= {
        "cta",
        "lung-glass",
    };
    static const std::string S_PRESET_VRT_PATH[S_PRESET_VRT_NUM] = {
        "../config/lut/3d/ct_cta.xml",
        "../config/lut/3d/ct_lung_glass.xml",
    };

    std::string target;
    for (int i = 0; i< S_PRESET_VRT_NUM; ++i) {
        if (context == S_PRESET_VRT_NAME[i]) {
            target = S_PRESET_VRT_PATH[i];
            break;
        }
    }
    if (target.empty()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "can't find preset VAT name: " << context << ".";
        return 0;
    }
    
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ColorTransFunc> color;
    std::shared_ptr<OpacityTransFunc> opacity;
    float ww, wl;
    RGBAUnit background;
    Material material;
    if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(target, color, opacity, ww, wl, background, material)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "load lut: " << target << " failed.";
        return 0;
    }

    std::map<unsigned int, std::shared_ptr<AppCell>> cells = controller->get_cells();
    for (auto it = cells.begin(); it != cells.end(); ++it) {
        std::shared_ptr<SceneBase> scene = it->second->get_scene();    
        APPCOMMON_CHECK_NULL_EXCEPTION(scene);
        std::shared_ptr<VRScene> vr_scene = std::dynamic_pointer_cast<VRScene>(scene);
        if (vr_scene) {
            if (vr_scene->get_mask_mode() == MASK_NONE) {
                vr_scene->set_window_level(ww, wl, 0);
                vr_scene->set_color_opacity(color, opacity, 0);
                vr_scene->set_material(material, 0);
            } else {
                const unsigned char MAIN_LABEL = 1;
                vr_scene->set_window_level(ww, wl, MAIN_LABEL);
                vr_scene->set_color_opacity(color, opacity, MAIN_LABEL);
                vr_scene->set_material(material, MAIN_LABEL);
            }
        }
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT OpSwitchPresetVRT.";
    return 0;
}

MED_IMG_END_NAMESPACE