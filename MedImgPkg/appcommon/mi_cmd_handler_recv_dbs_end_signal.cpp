#include "mi_cmd_handler_recv_dbs_end_signal.h"

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_common.h"

#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_message.pb.h"
#include "mi_app_common_util.h"
#include "mi_model_dbs_status.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerRecvDBSEndSignal::CmdHandlerRecvDBSEndSignal(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

CmdHandlerRecvDBSEndSignal::~CmdHandlerRecvDBSEndSignal() {

}

int CmdHandlerRecvDBSEndSignal::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN recveive DB server DICOM series cmd handler.";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<ModelDBSStatus> model_dbs_status = AppCommonUtil::get_model_dbs_status(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_dbs_status);
    model_dbs_status->set_success();

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT recveive DB server DICOM series cmd handler.";
    return CLIENT_QUIT_ID;
}

MED_IMG_END_NAMESPACE
