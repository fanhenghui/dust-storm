#include "mi_cmd_handler_be_fe_pacs_fetch.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_message.pb.h"
#include "mi_app_config.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerBE_FEPACSFetch::CmdHandlerBE_FEPACSFetch(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerBE_FEPACSFetch::~CmdHandlerBE_FEPACSFetch() {}

int CmdHandlerBE_FEPACSFetch::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS fetch";
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //check msg
    MsgString msg;
    if (!msg.ParseFromArray(buffer, dataheader.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse series message from FE PACS fetch failed.";
        MemShield shield(buffer);
        msg.Clear();
        return -1;
    }
    msg.Clear();
    
    //send message to DBS to fetch choosed DICOM series
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_PACS_FETCH;
    IPCPackage* package = new IPCPackage(header,buffer);
    if(0 != controller->get_client_proxy_dbs()->sync_send_data(package)) {
        delete package;
        MI_APPCOMMON_LOG(MI_ERROR) << "send to DB to fetch PACS failed.";
        return -1;
    }    

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS fetch";
    return 0;
}

MED_IMG_END_NAMESPACE