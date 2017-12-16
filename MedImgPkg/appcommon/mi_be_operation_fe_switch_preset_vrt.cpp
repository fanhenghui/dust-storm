#include "mi_be_operation_fe_switch_preset_vrt.h"

#include "io/mi_protobuf.h"

#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_opacity_transfer_function.h"
#include "renderalgo/mi_transfer_function_loader.h"
#include "renderalgo/mi_vr_scene.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFESwitchPresetVRT::BEOpFESwitchPresetVRT() {}

BEOpFESwitchPresetVRT::~BEOpFESwitchPresetVRT() {}

int BEOpFESwitchPresetVRT::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFESwitchPresetVRT.";
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse switch preset VRT message failed.";
        return -1;
    }

    const std::string context = msg.context();
    msg.Clear();
    
    static const int S_PRESET_VRT_NUM = 13;
    static const std::string S_PRESET_VRT_NAME[S_PRESET_VRT_NUM]= {
        "ct_calcification",
        "ct_carotids",
        "ct_clr_abd_aorta_1",
        "ct_clr_abd_aorta_2",
        "ct_clr_carotid_1",
        "ct_clr_carotid_2",
        "ct_color_vessel_gd",
        "ct_cta",
        "ct_cta_1",
        "ct_lung_glass",
        "ct_lung_glass_2",
        "ct_lung_bw",
        "ct_lung_2"
    };
    static const std::string S_PRESET_VRT_PATH[S_PRESET_VRT_NUM] = {
        "../config/lut/3d/ct_calcification.xml",
        "../config/lut/3d/ct_carotids.xml",
        "../config/lut/3d/ct_clr_abd_aorta_1.xml",
        "../config/lut/3d/ct_clr_abd_aorta_2.xml",
        "../config/lut/3d/ct_clr_carotid_1.xml",
        "../config/lut/3d/ct_clr_carotid_2.xml",
        "../config/lut/3d/ct_color_vessel_gd.xml",
        "../config/lut/3d/ct_cta.xml",
        "../config/lut/3d/ct_cta_1.xml",
        "../config/lut/3d/ct_lung_glass.xml",
        "../config/lut/3d/ct_lung_glass_2.xml",
        "../config/lut/3d/ct_lung_bw.xml",
        "../config/lut/3d/ct_lung_2.xml"
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
    
    std::shared_ptr<AppController> controller = get_controller<AppController>();
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
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFESwitchPresetVRT.";
    return 0;
}

MED_IMG_END_NAMESPACE