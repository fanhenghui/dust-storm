#include "mi_ai_server_controller.h"

#include "util/mi_ipc_client_proxy.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_config.h"
#include "appcommon/mi_operation_factory.h"

#include "mi_ai_server_thread_model.h"
#include "mi_ai_cmd_handler_operating.h"
#include "mi_ai_operation_evaluate.h"

MED_IMG_BEGIN_NAMESPACE

class ReadyEvent: public IEvent {
public:
    ReadyEvent(std::shared_ptr<AIServerThreadModel> thread_model):_thread_model(thread_model) {}
    virtual ~ReadyEvent() {};
    virtual void execute() {
        std::shared_ptr<AIServerThreadModel> thread_model = _thread_model.lock();
        if (nullptr == thread_model) {
            MI_AISERVER_LOG(MI_FATAL) << "thread model is null when run AI server connect event.";
        } else {
            IPCDataHeader header;
            header.msg_id = COMMAND_ID_AI_DB_OPERATION;
            header.msg_info1 = OPERATION_ID_DB_RECEIVE_AI_READY;
            header.data_len = 0;
            thread_model->async_send_data(new IPCPackage(header));
        }
    }

private:
    std::weak_ptr<AIServerThreadModel> _thread_model;
};

AIServerController::AIServerController() {
    _client_proxy.reset(new IPCClientProxy(UNIX)); 
    _client_proxy->set_path("/tmp/MIAIS");
    _thread_model.reset(new AIServerThreadModel());
    _thread_model->set_client_proxy(_client_proxy);
}

AIServerController::~AIServerController() {

}

void AIServerController::initialize() {
    //register cmd handler
    _client_proxy->register_command_handler(COMMAND_ID_DB_AI_OPERATION, 
        std::shared_ptr<CmdHandlerAIOperating>(new CmdHandlerAIOperating(shared_from_this())));

    OperationFactory::instance()->register_operation(OPERATION_ID_DB_REQUEST_AI_EVALUATION, 
    std::shared_ptr<AIOpEvaluate>(new AIOpEvaluate()));

    //register event for on connect to DB Server
    _client_proxy->register_on_connection_event(std::shared_ptr<ReadyEvent>(new ReadyEvent(_thread_model)));
}

void AIServerController::run() {
    _thread_model->start();
}

void AIServerController::finalize() {
    _thread_model->stop();
}

std::shared_ptr<AIServerThreadModel> AIServerController::get_thread_model() {
    return _thread_model;
}

std::shared_ptr<IPCClientProxy> AIServerController::get_client_proxy() {
    return _client_proxy;
}

MED_IMG_END_NAMESPACE