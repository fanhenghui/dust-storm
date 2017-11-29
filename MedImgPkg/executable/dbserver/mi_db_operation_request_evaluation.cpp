#include "mi_db_operation_request_evaluation.h"

#include "util/mi_ipc_server_proxy.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpRequestEvaluation::DBOpRequestEvaluation() {

}

DBOpRequestEvaluation::~DBOpRequestEvaluation() {

}

int DBOpRequestEvaluation::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);
    
    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_ais();

    IPCPackage* package = new IPCPackage(_header, _buffer);
    _buffer = nullptr;//move op buffer to IPC package

    if (0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send request infernce.(client disconnected)";
    }

    return 0;
}

MED_IMG_END_NAMESPACE