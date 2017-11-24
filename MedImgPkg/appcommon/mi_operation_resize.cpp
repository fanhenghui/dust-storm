#include "mi_operation_resize.h"

#include "io/mi_image_data.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"

#include "arithmetic/mi_ortho_camera.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_message.pb.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

OpResize::OpResize() {}

OpResize::~OpResize() {}

int OpResize::execute() {
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgResize msg;

    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        APPCOMMON_THROW_EXCEPTION("parse resize message failed!");
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

    return 0;
}

MED_IMG_END_NAMESPACE