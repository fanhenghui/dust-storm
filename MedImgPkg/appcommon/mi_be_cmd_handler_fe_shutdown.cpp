#include "mi_be_cmd_handler_fe_shutdown.h"

#include "mi_app_controller.h"
#include "util/mi_ipc_common.h"
#include "util/mi_memory_shield.h"

#include <arpa/inet.h>

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEShutdown::BECmdHandlerFEShutdown(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

BECmdHandlerFEShutdown::~BECmdHandlerFEShutdown() {

}

int BECmdHandlerFEShutdown::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
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
