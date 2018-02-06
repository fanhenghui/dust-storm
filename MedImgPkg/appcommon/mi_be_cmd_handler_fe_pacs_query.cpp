#include "mi_be_cmd_handler_fe_pacs_query.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_message.pb.h"
#include "io/mi_configure.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPACSQuery::BECmdHandlerFEPACSQuery(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEPACSQuery::~BECmdHandlerFEPACSQuery() {}

int BECmdHandlerFEPACSQuery::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEPACSQuery";

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //send message to DBS to query DICOM
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_PACS_QUERY;
    IPCPackage* package = new IPCPackage(header, buffer);
    if(0 != controller->get_client_proxy_dbs()->sync_send_data(package)) {
        delete package;
        MI_APPCOMMON_LOG(MI_ERROR) << "send to DB to query PACS failed.";
        return -1;
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEPACSQuery";
    return 0;
}

MED_IMG_END_NAMESPACE