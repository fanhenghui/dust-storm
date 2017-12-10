#include "mi_db_server_controller.h"

#include "util/mi_ipc_server_proxy.h"
#include "util/mi_operation_factory.h"

#include "io/mi_pacs_communicator.h"
#include "io/mi_db.h"
#include "io/mi_configure.h"

#include "appcommon/mi_app_common_define.h"

#include "mi_db_server_thread_model.h"
#include "mi_db_cmd_handler_be_operation.h"
#include "mi_db_operation_be_fetch_dicom.h"
#include "mi_db_operation_be_fetch_ai_evaluation.h"
#include "mi_db_operation_be_fetch_preprocess_mask.h"
#include "mi_db_operation_be_request_end.h"
#include "mi_db_operation_be_pacs_retrieve.h"
#include "mi_db_operation_be_pacs_fetch.h"
#include "mi_db_cmd_handler_ai_operation.h"
#include "mi_db_operation_ai_send_evaluation.h"
#include "mi_db_operation_ai_ready.h"
#include "mi_db_evaluatiion_dispatcher.h"
#include "mi_db_server_console_echo.h"

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
    _pacs_communicator.reset(new PACSCommunicator());
}

DBServerController::~DBServerController() {

}

void DBServerController::initialize() {
    //register cmd handler for BE 
    _server_proxy_be->register_command_handler(COMMAND_ID_DB_BE_OPERATION, 
        std::shared_ptr<DBCmdHandlerBEOperation>(new DBCmdHandlerBEOperation(shared_from_this())));
    //register operation
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_BE_FETCH_DICOM, 
        std::shared_ptr<DBOpBEFetchDICOM>(new DBOpBEFetchDICOM()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_BE_FETCH_AI_EVALUATION, 
        std::shared_ptr<DBOpBEFetchAIEvaluation>(new DBOpBEFetchAIEvaluation()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_BE_FETCH_PREPROCESS_MASK, 
        std::shared_ptr<DBOpBEFetchPreprocessMask>(new DBOpBEFetchPreprocessMask()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_BE_REQUEST_END, 
        std::shared_ptr<DBOpBERequestEnd>(new DBOpBERequestEnd()));
    
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_BE_PACS_RETRIEVE, 
        std::shared_ptr<DBOpBEPACSRetrieve>(new DBOpBEPACSRetrieve()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_BE_PACS_FETCH, 
        std::shared_ptr<DBOpBEPACSFetch>(new DBOpBEPACSFetch()));

    //register cmd handler for AI server
    _server_proxy_ais->register_command_handler(COMMAND_ID_DB_AI_OPERATION, 
        std::shared_ptr<DBCmdHandlerAIOperation>(new DBCmdHandlerAIOperation(shared_from_this())));

    OperationFactory::instance()->register_operation(OPERATION_ID_DB_AI_EVALUATION_RESULT, 
        std::shared_ptr<DBOpAISendEvaluation>(new DBOpAISendEvaluation()));
    OperationFactory::instance()->register_operation(OPERATION_ID_DB_AI_READY, 
        std::shared_ptr<DBOpAIReady>(new DBOpAIReady()));

    //connect db
    std::string ip_port,user,pwd,db_name;
    Configure::instance()->get_db_info(ip_port, user, pwd, db_name);
    if(0 != _db->connect(user, ip_port, pwd, db_name) ) {
        MI_DBSERVER_LOG(MI_FATAL) << "connect to db failed.";
        DBSERVER_THROW_EXCEPTION("connect to db failed.");
    }

    //set controller to dispatcher
    _evaluation_dispatcher->set_controller(shared_from_this());

    //create console echo
    std::shared_ptr<DBServerConsoleEcho> console_echo(new DBServerConsoleEcho(shared_from_this()));
    _thread_model->set_console_echo(console_echo);

    //connect PACS
    std::string server_ae_title,server_host,client_ae_title;
    unsigned short server_port(0), client_port(0);
    Configure::instance()->get_pacs_info(server_ae_title, server_host, server_port, client_ae_title, client_port);
    if(-1 == _pacs_communicator->connect(server_ae_title, server_host, server_port, client_ae_title, client_port)) {
        MI_DBSERVER_LOG(MI_FATAL) << "Connect to PACS {AET: " << server_ae_title << "; URL: " << server_host << ":" << server_port << "} failed.";
        //Start with disconnect.
        //DBSERVER_THROW_EXCEPTION("connect to PACS failed.");
    }
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

std::shared_ptr<PACSCommunicator> DBServerController::get_pacs_communicator() {
    return _pacs_communicator;
}

void DBServerController::set_ais_client(unsigned int client) {
    _ais_client = client;
}

unsigned int DBServerController::get_ais_client() const {
    return _ais_client;
}

MED_IMG_END_NAMESPACE