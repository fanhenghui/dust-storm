#include "mi_db_operation_be_request_end.h"

#include "util/mi_ipc_server_proxy.h"

#include "appcommon/mi_app_common_define.h"

#include "mi_db_server_controller.h"


MED_IMG_BEGIN_NAMESPACE

DBOpBERequestEnd::DBOpBERequestEnd() {

}

DBOpBERequestEnd::~DBOpBERequestEnd() {

}

int DBOpBERequestEnd::execute() {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBERequestEnd.";

    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);

    IPCDataHeader header;
    header.msg_id = COMMAND_ID_BE_DB_SEND_END;
    header.receiver = _header.receiver;

    server_proxy->async_send_data(new IPCPackage(header));

    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBERequestEnd.";

    return 0;
}

MED_IMG_END_NAMESPACE