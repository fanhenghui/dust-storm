#include "mi_ai_server_controller.h"

#include "util/mi_ipc_client_proxy.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_config.h"
#include "appcommon/mi_operation_factory.h"

#include "mi_ai_server_thread_model.h"
#include "mi_ai_cmd_handler_operating.h"


MED_IMG_BEGIN_NAMESPACE

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
    // _server_proxy_be->register_command_handler(COMMAND_ID_BE_DB_OPERATION, 
    //     std::shared_ptr<CmdHandlerDBBEOperating>(new CmdHandlerDBBEOperating(shared_from_this())));
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