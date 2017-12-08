#include "mi_db_operation_fetch_ai_evaluation.h"

#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"
#include "mi_db_evaluatiion_dispatcher.h"

MED_IMG_BEGIN_NAMESPACE

DBOpFetchAIEvaluation::DBOpFetchAIEvaluation() {

}

DBOpFetchAIEvaluation::~DBOpFetchAIEvaluation() {

}

int DBOpFetchAIEvaluation::execute() {
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

    if (-1 == dispatcher->request_evaluation(_header.receiver, series_id) ) {
        MI_DBSERVER_LOG(MI_ERROR) << "rquest evaluation failed.";
        return -1;
    }

    return 0;
}

MED_IMG_END_NAMESPACE