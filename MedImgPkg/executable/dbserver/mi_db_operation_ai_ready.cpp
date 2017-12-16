#include "mi_db_operation_ai_ready.h"

#include "util/mi_ipc_server_proxy.h"

#include "mi_db_server_controller.h"
#include "mi_db_server_common.h"

MED_IMG_BEGIN_NAMESPACE

DBOpAIReady::DBOpAIReady() {

}

DBOpAIReady::~DBOpAIReady() {

}

int DBOpAIReady::execute() {    
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpAIReady.";

    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    controller->set_ais_client(_header.receiver);
    if (controller->get_ais_client() == 0) {
        MI_DBSERVER_LOG(MI_FATAL) << "AIS client ID is 0.";
        return -1;
    } else {
        MI_DBSERVER_LOG(MI_INFO) << "AIS ready.";
        MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpAIReady.";
        return 0;
    }
}

MED_IMG_END_NAMESPACE