#include "mi_be_cmd_handler_fe_shutdown.h"

#include "util/mi_ipc_common.h"
#include "util/mi_memory_shield.h"

#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEShutdown::BECmdHandlerFEShutdown(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

BECmdHandlerFEShutdown::~BECmdHandlerFEShutdown() {

}

int BECmdHandlerFEShutdown::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEShutdown";

    MemShield shield(buffer);

    //TODO return shutdown ID
    const int quit_id = CLIENT_QUIT_ID;
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEShutdown";
    return quit_id;
}


MED_IMG_END_NAMESPACE
