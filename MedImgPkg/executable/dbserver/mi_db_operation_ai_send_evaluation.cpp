#include "mi_db_operation_ai_send_evaluation.h"

#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"
#include "mi_db_evaluatiion_dispatcher.h"

MED_IMG_BEGIN_NAMESPACE

DBOpAISendEvaluation::DBOpAISendEvaluation() {

}

DBOpAISendEvaluation::~DBOpAISendEvaluation() {

}

int DBOpAISendEvaluation::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<DBEvaluationDispatcher> dispatcher = controller->get_evaluation_dispatcher();
    DBSERVER_CHECK_NULL_EXCEPTION(dispatcher);

    MsgEvaluationResponse msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        DBSERVER_THROW_EXCEPTION("parse evaluation response message failed!");
    }
    if (-1 == dispatcher->receive_evaluation(&msg)) {
        MI_DBSERVER_LOG(MI_DEBUG) << "receive evaluation failed."; 
        msg.Clear();
        return -1;
    }
    msg.Clear();
    

    MI_DBSERVER_LOG(MI_DEBUG) << "receive AIS result and send to BE.";

    return 0;
}

MED_IMG_END_NAMESPACE