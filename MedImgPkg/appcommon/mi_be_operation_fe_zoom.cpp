#include "mi_be_operation_fe_zoom.h"

#include "io/mi_image_data.h"
#include "io/mi_protobuf.h"

#include "arithmetic/mi_ortho_camera.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFEZoom::BEOpFEZoom() {}

BEOpFEZoom::~BEOpFEZoom() {}

int BEOpFEZoom::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFEZoom.";

    const unsigned int cell_id = _header.cell_id;
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgMouse msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse zoom mouse message failed.";
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

    scene->zoom(Point2(pre_x, pre_y), Point2(cur_x, cur_y));

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFEZoom.";
    return 0;
}

MED_IMG_END_NAMESPACE