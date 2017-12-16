#include "mi_db_operation_ai_send_evaluation.h"

#include "io/mi_protobuf.h"

#include "mi_db_server_controller.h"
#include "mi_db_evaluatiion_dispatcher.h"

MED_IMG_BEGIN_NAMESPACE

DBOpAISendEvaluation::DBOpAISendEvaluation() {

}

DBOpAISendEvaluation::~DBOpAISendEvaluation() {

}

int DBOpAISendEvaluation::execute() {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpAISendEvaluation.";
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<DBEvaluationDispatcher> dispatcher = controller->get_evaluation_dispatcher();
    DBSERVER_CHECK_NULL_EXCEPTION(dispatcher);

    MsgEvaluationResponse msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse evaluation message send by AI server failed.";
        return -1;
    }

    if (-1 == dispatcher->receive_evaluation(&msg)) {
        MI_DBSERVER_LOG(MI_DEBUG) << "receive evaluation failed."; 
        msg.Clear();
        return -1;
    }
    msg.Clear();
    

    MI_DBSERVER_LOG(MI_DEBUG) << "receive AIS result and send to BE.";
    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpAISendEvaluation.";
    return 0;
}

MED_IMG_END_NAMESPACE