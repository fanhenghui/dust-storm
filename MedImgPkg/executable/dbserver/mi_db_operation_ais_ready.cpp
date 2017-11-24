#include "mi_db_operation_ais_ready.h"

#include "util/mi_ipc_server_proxy.h"

#include "mi_db_server_controller.h"
#include "mi_db_server_common.h"

MED_IMG_BEGIN_NAMESPACE

DBOpAISReady::DBOpAISReady() {

}

DBOpAISReady::~DBOpAISReady() {

}

int DBOpAISReady::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);
    
    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    controller->set_ais_socket_id(_header.receiver);
    if (controller->get_ais_socket_id() == 0) {
        MI_DBSERVER_LOG(MI_FATAL) << "AIS socket ID is 0.";
        return -1;
    } else {
        MI_DBSERVER_LOG(MI_FATAL) << "AIS ready.";
        return 0;
    }
}

MED_IMG_END_NAMESPACE