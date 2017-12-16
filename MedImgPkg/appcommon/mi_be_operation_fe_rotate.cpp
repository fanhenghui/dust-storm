#include "mi_be_operation_fe_rotate.h"

#include "arithmetic/mi_matrix4.h"
#include "arithmetic/mi_quat4.h"
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

BEOpFERotate::BEOpFERotate() {}

BEOpFERotate::~BEOpFERotate() {}

int BEOpFERotate::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFERotate.";
    const unsigned int cell_id = _header.cell_id;
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    APPCOMMON_CHECK_NULL_EXCEPTION(cell);

    std::shared_ptr<SceneBase> scene = cell->get_scene();
    APPCOMMON_CHECK_NULL_EXCEPTION(scene);

    MsgMouse msg;
    //if (msg.ParseFromArray(_buffer, _header.data_len)) {
    if (0 == protobuf_parse(_buffer, _header.data_len, msg)) {
        const float pre_x = msg.pre().x();
        const float pre_y = msg.pre().y();
        const float cur_x = msg.cur().x();
        const float cur_y = msg.cur().y();
        msg.Clear();
    
        scene->rotate(Point2(pre_x, pre_y), Point2(cur_x, cur_y));
    } else {
        //second change to parse to rotation message
        MsgRotation msg;    
        if(0 == protobuf_parse(_buffer, _header.data_len, msg)) {
            const double angle = static_cast<double>(msg.angle());
            const double axis_x = static_cast<double>(msg.axis_x());
            const double axis_y = static_cast<double>(msg.axis_y());
            const double axis_z = static_cast<double>(msg.axis_z());
            Quat4 quat(angle,Vector3(axis_x,axis_y,axis_z));
            scene->get_camera()->rotate(quat);
            scene->set_dirty(true);   
        } else {
            MI_APPCOMMON_LOG(MI_ERROR) << "parse rotation message failed.";
            msg.Clear();
        }
    }
    
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFERotate";
    return 0;
}

MED_IMG_END_NAMESPACE