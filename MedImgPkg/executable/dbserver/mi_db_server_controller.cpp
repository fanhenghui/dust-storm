#include "mi_db_server_controller.h"

#include "util/mi_ipc_server_proxy.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_config.h"
#include "appcommon/mi_operation_factory.h"

#include "mi_db_server_thread_model.h"

#include "mi_db_cmd_handler_be_operating.h"
#include "mi_db_operation_query_dicom.h"
#include "mi_db_operation_query_ai_annotation.h"
#include "mi_db_operation_query_preprocess_mask.h"
#include "mi_db_operation_query_end.h"

#include "mi_db_cmd_handler_ais_operating.h"
#include "mi_db_operation_receive_evaluation.h"
#include "mi_db_operation_receive_ai_ready.h"
#include "mi_db_evaluatiion_dispatcher.h"

MED_IMG_BEGIN_NAMESPACE

DBServerController::DBServerController() {
    _server_proxy_be.reset(new IPCServerProxy(INET)); 
    _server_proxy_be->set_server_address("8888");
    _server_proxy_ais.reset(new IPCServerProxy(UNIX));
    _server_proxy_ais->set_path("/tmp/MIAIS");
    _thread_model.reset(new DBServerThreadModel());
    _thread_model->set_server_proxy_be(_server_proxy_be);
    _thread_model->set_server_proxy_ais(_server_proxy_ais);
    _db.reset(new DB());
    _evaluation_dispatcher.reset(new DBEvaluationDispatcher());
}

DBServerController::~DBServerController() {

}

void DBServerController::initialize() {
    //register cmd handler for BE 
    _server_proxy_be->register_command_handler(COMMAND_ID_BE_DB_OPERATION, 
        std::shared_ptr<CmdHandlerDBBEOperating>(new CmdHandlerDBBEOperating(shared_from_this())));
    //register operation
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_QUERY_DICOM, 
        std::shared_ptr<DBOpQueryDICOM>(new DBOpQueryDICOM()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_QUERY_AI_ANNOTATION, 
        std::shared_ptr<DBOpQueryAIAnnotation>(new DBOpQueryAIAnnotation()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_QUERY_PREPROCESS_MASK, 
        std::shared_ptr<DBOpQueryPreprocessMask>(new DBOpQueryPreprocessMask()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_QUERY_END, 
        std::shared_ptr<DBOpQueryEnd>(new DBOpQueryEnd()));


    //register cmd handler for AI server
    _server_proxy_ais->register_command_handler(COMMAND_ID_AI_DB_OPERATION, 
        std::shared_ptr<CmdHandlerDBAISOperating>(new CmdHandlerDBAISOperating(shared_from_this())));

    OperationFactory::instance()->register_operation(OPERATION_ID_DB_RECEIVE_AI_EVALUATION, 
        std::shared_ptr<DBOpReceiveEvaluation>(new DBOpReceiveEvaluation()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_RECEIVE_AI_READY, 
        std::shared_ptr<DBOpReceiveAIReady>(new DBOpReceiveAIReady()));

    //connect db
    std::string ip_port,user,pwd,db_name;
    AppConfig::instance()->get_db_info(ip_port, user, pwd, db_name);
    if(0 != _db->connect(user, ip_port, pwd, db_name) ) {
        MI_DBSERVER_LOG(MI_FATAL) << "connect to db failed.";
        DBSERVER_THROW_EXCEPTION("connect to db failed.");
    }

    //set controller to dispatcher
    _evaluation_dispatcher->set_controller(shared_from_this());
}

void DBServerController::run() {
    _thread_model->start();
}

void DBServerController::finalize() {
    _thread_model->stop();
    _db->disconnect();
}

std::shared_ptr<DBServerThreadModel> DBServerController::get_thread_model() {
    return _thread_model;
}

std::shared_ptr<IPCServerProxy> DBServerController::get_server_proxy_be() {
    return _server_proxy_be;
}

std::shared_ptr<IPCServerProxy> DBServerController::get_server_proxy_ais() {
    return _server_proxy_ais;
}

std::shared_ptr<DB> DBServerController::get_db() {
    return _db;
}

std::shared_ptr<DBEvaluationDispatcher> DBServerController::get_evaluation_dispatcher() {
    return _evaluation_dispatcher;
}

void DBServerController::set_ais_client(unsigned int client) {
    _ais_client = client;
}

unsigned int DBServerController::get_ais_client() const {
    return _ais_client;
}

MED_IMG_END_NAMESPACE