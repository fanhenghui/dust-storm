#include "mi_cmd_handler_be_fe_shutdown.h"

#include "mi_app_controller.h"
#include "util/mi_ipc_common.h"
#include "util/mi_memory_shield.h"

#include <arpa/inet.h>

MED_IMG_BEGIN_NAMESPACE

CmdHandlerBE_FEShutdown::CmdHandlerBE_FEShutdown(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

CmdHandlerBE_FEShutdown::~CmdHandlerBE_FEShutdown() {

}

int CmdHandlerBE_FEShutdown::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();

    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    //TODO return shutdown ID
    const int quit_id = CLIENT_QUIT_ID;
    return quit_id;
}


MED_IMG_END_NAMESPACE
