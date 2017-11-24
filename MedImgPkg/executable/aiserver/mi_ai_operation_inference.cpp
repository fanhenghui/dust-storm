#include "mi_ai_operation_inference.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_message.pb.h"

#include "mi_ai_server_controller.h"
#include "mi_ai_server_thread_model.h"

MED_IMG_BEGIN_NAMESPACE

AIOpInference::AIOpInference() {

}

AIOpInference::~AIOpInference() {

}

int AIOpInference::execute() {
    AISERVER_CHECK_NULL_EXCEPTION(_buffer);
    
    std::shared_ptr<AIServerController> controller  = get_controller<AIServerController>();
    AISERVER_CHECK_NULL_EXCEPTION(controller);

    //TODO do calcualte 

    //TODO async push package to queue

    return 0;
}

MED_IMG_END_NAMESPACE