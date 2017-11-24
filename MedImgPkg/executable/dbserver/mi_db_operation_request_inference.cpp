#include "mi_db_operation_request_inference.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpRequestInference::DBOpRequestInference() {

}

DBOpRequestInference::~DBOpRequestInference() {

}

int DBOpRequestInference::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);
    
    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_ais();

    const unsigned int receiver = controller->get_ais_socket_id();

    IPCDataHeader header;
    header.receiver = receiver;
    header.data_len = _header.data_len;
    header.msg_id = COMMAND_ID_DB_AI_OPERATION;
    header.msg_info1 = OPERATION_ID_DB_REQUEST_AI_INFERENCE;
    char* buffer_cp = new char[_header.data_len];// the buffer is desearlized by query AI annotation op(MsgInferenceRequest)
    memcpy(buffer_cp, _buffer, _header.data_len);
    IPCPackage* package = new IPCPackage(header, buffer_cp);
    if (0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send dcm to client failed.(client disconnected)";
    }

    return 0;
}

MED_IMG_END_NAMESPACE