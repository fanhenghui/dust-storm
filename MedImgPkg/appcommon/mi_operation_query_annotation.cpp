#include "mi_operation_query_annotation.h"

#include "util/mi_ipc_client_proxy.h"
#include "arithmetic/mi_circle.h"

#include "renderalgo/mi_mask_label_store.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_annotation_calculator.h"
#include "renderalgo/mi_volume_infos.h"

#include "mi_model_annotation.h"
#include "mi_model_crosshair.h"
#include "mi_model_dbs_status.h"
#include "mi_app_cell.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_none_image_item.h"
#include "mi_app_none_image.h"
#include "mi_message.pb.h"
#include "mi_app_common_util.h"
#include "mi_app_config.h"

MED_IMG_BEGIN_NAMESPACE

OpQueryAnnotation::OpQueryAnnotation() {}

OpQueryAnnotation::~OpQueryAnnotation() {}

int OpQueryAnnotation::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN OpQueryAnnotation.";
    if (_buffer == nullptr || _header.data_len == 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "incompleted annotation request message from BE.";
        return -1;
    }

    MsgAnnotationQuery msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse annotation request message from BE failed.";
        return -1;
    }
    const int role = msg.role();
    const std::string username = msg.username();
    const std::string serise_id = msg.series_uid();
    msg.Clear();

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    if (role == 0) {
        IPCDataHeader post_header;
        char* post_data = nullptr;

        post_header.msg_id = COMMAND_ID_BE_DB_OPERATION;
        post_header.op_id = OPERATION_ID_DB_QUERY_AI_ANNOTATION;

        MsgString msgSeries;
        msgSeries.set_context(serise_id);
        int post_size = msgSeries.ByteSize();
        post_data = new char[post_size];
        if (!msgSeries.SerializeToArray(post_data, post_size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "create series message failed. ";
            return -1;
        }
        post_header.data_len = post_size;
        std::vector<IPCPackage*> pkgs;
        pkgs.push_back(new IPCPackage(post_header,post_data));
        controller->get_client_proxy_dbs()->sync_send_data(pkgs);

    } else {
        //TODO query from annotation table
    }



    MI_APPCOMMON_LOG(MI_TRACE) << "OUT OpQueryAnnotation.";
    return 0;
}

MED_IMG_END_NAMESPACE