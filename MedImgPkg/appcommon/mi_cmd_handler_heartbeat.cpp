#include "mi_cmd_handler_heartbeat.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerHeartbeat::CmdHandlerHeartbeat(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerHeartbeat::~CmdHandlerHeartbeat() {}

int CmdHandlerHeartbeat::handle_command(const IPCDataHeader &ipcheader, char *buffer) {
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }
    
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_FE_BE_HEARTBEAT;

    MI_APPCOMMON_LOG(MI_INFO) << "BE reveive heartbeat.";
    controller->get_client_proxy()->sync_send_data(header, nullptr);

    return 0;
}

MED_IMG_END_NAMESPACE
