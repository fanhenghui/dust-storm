#include "mi_cmd_handler_be_db_send_error.h"

#include "util/mi_memory_shield.h"

#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_message.pb.h"
#include "mi_model_dbs_status.h"
#include "mi_app_common_define.h"
#include "mi_app_common_util.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerBE_DBSendError::CmdHandlerBE_DBSendError(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

CmdHandlerBE_DBSendError::~CmdHandlerBE_DBSendError() {

}

int CmdHandlerBE_DBSendError::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN recveive DB server error cmd handler.";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelDBSStatus> model_dbs_status = AppCommonUtil::get_model_dbs_status(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_dbs_status);

    MsgString msg;
    if (!msg.ParseFromArray(buffer, ipcheader.data_len)) {
        model_dbs_status->push_error_info("parse recv dbs error message failed.");
        return -1; 
    }

    const std::string err_msg = msg.context();
    msg.Clear();
    model_dbs_status->push_error_info(err_msg);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT recveive DB server error cmd handler.";
    return 0;
}


MED_IMG_END_NAMESPACE
