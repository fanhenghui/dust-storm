#include "mi_cmd_handler_pacs_fetch_response.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_message.pb.h"
#include "mi_app_config.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerPACSFetchResponse::CmdHandlerPACSFetchResponse(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerPACSFetchResponse::~CmdHandlerPACSFetchResponse() {}

int CmdHandlerPACSFetchResponse::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS fetch response";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //receive DBS's fetch response, and create operation to BE queue to notify FE to update series status(in PACS table)

    

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS fetch response";
    return 0;
}

MED_IMG_END_NAMESPACE