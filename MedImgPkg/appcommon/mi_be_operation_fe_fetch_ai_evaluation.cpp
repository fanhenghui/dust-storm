#include "mi_be_operation_fe_fetch_ai_evaluation.h"

#include "util/mi_ipc_client_proxy.h"

#include "arithmetic/mi_circle.h"

#include "io/mi_protobuf.h"
#include "io/mi_configure.h"

#include "renderalgo/mi_mask_label_store.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_volume_infos.h"

#include "mi_annotation_calculator.h"
#include "mi_model_annotation.h"
#include "mi_model_crosshair.h"
#include "mi_model_dbs_status.h"
#include "mi_app_cell.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_none_image_item.h"
#include "mi_app_none_image.h"
#include "mi_app_common_util.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFEFetchAIEvaluation::BEOpFEFetchAIEvaluation() {}

BEOpFEFetchAIEvaluation::~BEOpFEFetchAIEvaluation() {}

int BEOpFEFetchAIEvaluation::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFEFetchAIEvaluation.";
    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

    MsgAnnotationQuery msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse evaluation request message from BE failed.";
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
        post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
        post_header.op_id = OPERATION_ID_DB_BE_FETCH_AI_EVALUATION;

        MsgString msg_series;
        msg_series.set_context(serise_id);
        char* post_data = nullptr;
        int post_size = 0;
        if (0 != protobuf_serialize(msg_series, post_data, post_size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "create series message failed. ";
            return -1;
        }
        msg_series.Clear();
        
        post_header.data_len = post_size;
        std::vector<IPCPackage*> pkgs;
        pkgs.push_back(new IPCPackage(post_header,post_data));
        controller->get_client_proxy_dbs()->sync_send_data(pkgs);

    } else {
        //TODO query from annotation table
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFEFetchAIEvaluation.";
    return 0;
}

MED_IMG_END_NAMESPACE