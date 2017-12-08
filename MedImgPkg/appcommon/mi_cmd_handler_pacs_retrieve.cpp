#include "mi_cmd_handler_pacs_retrieve.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_message.pb.h"
#include "mi_app_config.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerPACSRetrieve::CmdHandlerPACSRetrieve(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerPACSRetrieve::~CmdHandlerPACSRetrieve() {}

int CmdHandlerPACSRetrieve::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS retrieve";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //send message to DBS to retrive all DICOM series
    std::vector<IPCPackage*> packages;
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_PACS_RETRIEVE;
    packages.push_back(new IPCPackage(header));
    controller->get_client_proxy_dbs()->sync_send_data(packages);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS retrieve";
    return 0;
}

MED_IMG_END_NAMESPACE