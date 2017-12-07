#include "mi_db_operation_pacs_fetch.h"

#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpPACSFetch::DBOpPACSFetch() {

}

DBOpPACSFetch::~DBOpPACSFetch() {

}

int DBOpPACSFetch::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        DBSERVER_THROW_EXCEPTION("parse series message failed!");
    }
    const std::string series_id = msg.context();
    msg.Clear();
    
    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<DBEvaluationDispatcher> dispatcher = controller->get_evaluation_dispatcher();
    DBSERVER_CHECK_NULL_EXCEPTION(dispatcher);


    return 0;
}

MED_IMG_END_NAMESPACE