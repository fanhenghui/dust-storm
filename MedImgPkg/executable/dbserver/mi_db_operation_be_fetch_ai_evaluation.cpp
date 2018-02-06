#include "mi_db_operation_be_fetch_ai_evaluation.h"

#include "io/mi_protobuf.h"

#include "mi_db_server_controller.h"
#include "mi_db_evaluatiion_dispatcher.h"

MED_IMG_BEGIN_NAMESPACE

DBOpBEFetchAIEvaluation::DBOpBEFetchAIEvaluation() {

}

DBOpBEFetchAIEvaluation::~DBOpBEFetchAIEvaluation() {

}

int DBOpBEFetchAIEvaluation::execute() {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBEFetchAIEvaluation.";
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    MsgEvaluationRetrieveKey msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse fetch evaluation message send by BE failed.";
        return -1;
    }
    
    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<DBEvaluationDispatcher> dispatcher = controller->get_evaluation_dispatcher();
    DBSERVER_CHECK_NULL_EXCEPTION(dispatcher);

    if (-1 == dispatcher->request_evaluation(_header.receiver, &msg) ) {
        MI_DBSERVER_LOG(MI_ERROR) << "rquest evaluation failed.";
        msg.Clear();
        return -1;
    }
    msg.Clear();

    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBEFetchAIEvaluation.";
    return 0;
}

MED_IMG_END_NAMESPACE