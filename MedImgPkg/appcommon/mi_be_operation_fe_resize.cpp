#include "mi_be_operation_fe_resize.h"

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

BEOpFEResize::BEOpFEResize() {}

BEOpFEResize::~BEOpFEResize() {}

int BEOpFEResize::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFEResize.";
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgResize msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse resize message failed.";
        return -1;
    }

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    for (int i = 0; i < msg.cells_size(); i++) {
        const MsgCellInfo& cell_info = msg.cells(i);
        const unsigned int cell_id = (unsigned int)cell_info.id();
        const int width = cell_info.width();
        const int height = cell_info.height();

        std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
        APPCOMMON_CHECK_NULL_EXCEPTION(cell);
        std::shared_ptr<SceneBase> scene = cell->get_scene();
        scene->set_display_size(width, height);
        scene->set_downsample(false);

        MI_APPCOMMON_LOG(MI_INFO) << "resize cell (id: " << cell_id << " width: " << width << " height: " << height << ")";
    }
    msg.Clear();

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFEResize.";
    return 0;
}

MED_IMG_END_NAMESPACE