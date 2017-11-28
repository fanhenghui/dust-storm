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

    //二次加工，将op的header和buffer转化成IPC的header 和 buffer并发送给AI Server
    //之所以放到队列里而不直接在DBS的BE cmd handler里执行，是为了在同一条线程中调度计算（请求和回复都是在AI的Op线程中）
    const unsigned int receiver = controller->get_ais_socket_id();

    IPCDataHeader header;
    header.receiver = receiver;
    header.data_len = _header.data_len;
    header.msg_id = COMMAND_ID_DB_AI_OPERATION;
    header.msg_info1 = OPERATION_ID_DB_REQUEST_AI_INFERENCE;
    IPCPackage* package = new IPCPackage(header, _buffer);
    _buffer = nullptr;//move op buffer to IPC package

    if (0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send request infernce.(client disconnected)";
    }

    return 0;
}

MED_IMG_END_NAMESPACE