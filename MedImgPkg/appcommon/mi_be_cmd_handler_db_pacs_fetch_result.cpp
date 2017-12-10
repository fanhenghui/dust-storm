#include "mi_be_cmd_handler_db_pacs_fetch_result.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_configure.h"
#include "io/mi_message.pb.h"

#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerDBPACSFetchResult::BECmdHandlerDBPACSFetchResult(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerDBPACSFetchResult::~BECmdHandlerDBPACSFetchResult() {}

int BECmdHandlerDBPACSFetchResult::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS fetch response";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //receive DBS's fetch response, and create operation to BE queue to notify FE to update series status(in PACS table)

    MsgString msg;
    if (!msg.ParseFromArray(buffer, dataheader.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse DB PACS fetch response message failed.";
        msg.Clear();
        return -1;
    }
    const std::string series_id = msg.context();
    msg.Clear();
    
    MI_APPCOMMON_LOG(MI_DEBUG) << "fetch series: " << series_id << " done.";

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS fetch response";
    return 0;
}

MED_IMG_END_NAMESPACE