#include "mi_be_operation_fe_downsample.h"

#include "io/mi_image_data.h"
#include "io/mi_message.pb.h"

#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_scene_base.h"

#include "mi_app_cell.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFEDownsample::BEOpFEDownsample() {}

BEOpFEDownsample::~BEOpFEDownsample() {}

int BEOpFEDownsample::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFEDownsample.";

    const unsigned int cell_id = _header.cell_id;
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);
    MsgFlag msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        APPCOMMON_THROW_EXCEPTION("parse mouse message failed!");
    }

    const bool flag = msg.flag();
    msg.Clear();
    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    APPCOMMON_CHECK_NULL_EXCEPTION(cell);

    std::shared_ptr<SceneBase> scene = cell->get_scene();
    APPCOMMON_CHECK_NULL_EXCEPTION(scene);

    scene->set_downsample(flag);
    if (!flag) {
        scene->set_dirty(true); // HD repaint
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFEDownsample.";
    return 0;
}

MED_IMG_END_NAMESPACE