#include "mi_cmd_handler_pacs_fetch.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_message.pb.h"
#include "mi_app_config.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerPACSFetch::CmdHandlerPACSFetch(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerPACSFetch::~CmdHandlerPACSFetch() {}

int CmdHandlerPACSFetch::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS fetch";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //send message to DBS to fetch choosed DICOM series

    

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS fetch";
    return 0;
}

MED_IMG_END_NAMESPACE