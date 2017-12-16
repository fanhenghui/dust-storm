#include "mi_be_operation_fe_switch_preset_windowing.h"

#include "arithmetic/mi_vector2f.h"
#include "io/mi_protobuf.h"
#include "renderalgo/mi_mpr_scene.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFESwitchPresetWindowing::BEOpFESwitchPresetWindowing() {}

BEOpFESwitchPresetWindowing::~BEOpFESwitchPresetWindowing() {}

int BEOpFESwitchPresetWindowing::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFESwitchPresetWindowing.";
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse switch preset windowing message failed!.";
        return -1;
    }
    const std::string context = msg.context();
    msg.Clear();
    
    static const int S_PRESET_WL_NUM = 6;
    static const std::string S_PRESET_WL_NAME[S_PRESET_WL_NUM]= {
        "Abdomen",
        "Lung",
        "Brain",
        "Angio",
        "Bone",
        "Chest"
    };
    static const Vector2f S_PRESET_WL[S_PRESET_WL_NUM] = {
        Vector2f(400,60),
        Vector2f(1500,-400),
        Vector2f(80,40),
        Vector2f(600,300),
        Vector2f(1500,300),
        Vector2f(400,40)
    };

    int target = -1;
    for (int i = 0; i< S_PRESET_WL_NUM; ++i) {
        if (context == S_PRESET_WL_NAME[i]) {
            target = i;
            break;
        }
    }
    if (target == -1) {
        MI_APPCOMMON_LOG(MI_ERROR) << "can't find preset WL name: " << context << ".";
        return 0;
    }
    const float ww = S_PRESET_WL[target].get_x();
    const float wl = S_PRESET_WL[target].get_y();
    
    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::map<unsigned int, std::shared_ptr<AppCell>> cells = controller->get_cells();
    for (auto it = cells.begin(); it != cells.end(); ++it) {
        std::shared_ptr<SceneBase> scene = it->second->get_scene();    
        APPCOMMON_CHECK_NULL_EXCEPTION(scene);
        std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
        if (mpr_scene) {
            mpr_scene->set_global_window_level(ww, wl);
            mpr_scene->set_dirty(true);
        }
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFESwitchPresetWindowing.";
    return 0;
}

MED_IMG_END_NAMESPACE