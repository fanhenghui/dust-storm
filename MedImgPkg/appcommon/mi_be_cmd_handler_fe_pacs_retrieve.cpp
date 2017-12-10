#include "mi_be_cmd_handler_fe_pacs_retrieve.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_message.pb.h"
#include "mi_app_config.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPACSRetrieve::BECmdHandlerFEPACSRetrieve(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEPACSRetrieve::~BECmdHandlerFEPACSRetrieve() {}

int BECmdHandlerFEPACSRetrieve::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS retrieve";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //send message to DBS to retrive all DICOM series
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_PACS_RETRIEVE;
    IPCPackage* package = new IPCPackage(header);
    if(0 != controller->get_client_proxy_dbs()->sync_send_data(package)) {
        delete package;
        MI_APPCOMMON_LOG(MI_ERROR) << "send to DB to retrieve PACS failed.";
        return -1;
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS retrieve";
    return 0;
}

MED_IMG_END_NAMESPACE