#include "mi_be_cmd_handler_fe_heartbeat.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEHeartbeat::BECmdHandlerFEHeartbeat(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEHeartbeat::~BECmdHandlerFEHeartbeat() {}

int BECmdHandlerFEHeartbeat::handle_command(const IPCDataHeader &ipcheader, char *buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEHeartbeat";

    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_FE_BE_HEARTBEAT;

    MI_APPCOMMON_LOG(MI_INFO) << "BE reveive heartbeat.";
    controller->get_client_proxy()->sync_send_data(header, nullptr);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEHeartbeat";
    return 0;
}

MED_IMG_END_NAMESPACE
