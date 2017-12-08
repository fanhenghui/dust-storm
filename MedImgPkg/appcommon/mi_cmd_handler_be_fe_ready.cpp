#include "mi_cmd_handler_be_fe_ready.h"

#include <iostream>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerBE_FEReady::CmdHandlerBE_FEReady(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerBE_FEReady::~CmdHandlerBE_FEReady() {}

int CmdHandlerBE_FEReady::handle_command(const IPCDataHeader& ipcheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN ready cmd handler.";
    MemShield shield(buffer);

    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    IPCDataHeader header;
    header.msg_id = COMMAND_ID_FE_BE_READY;

    MI_APPCOMMON_LOG(MI_INFO) << "sending ready to FE.";
    controller->get_client_proxy()->sync_send_data(header, nullptr);
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT ready cmd handler.";
    return 0;
}

MED_IMG_END_NAMESPACE
