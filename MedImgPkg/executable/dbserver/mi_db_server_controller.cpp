#include "mi_db_server_controller.h"

#include "util/mi_ipc_server_proxy.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_config.h"

#include "mi_db_server_thread_model.h"
#include "mi_db_cmd_handler_operating.h"


MED_IMG_BEGIN_NAMESPACE

DBServerController::DBServerController() {
    _server_proxy.reset(new IPCServerProxy(INET)); 
    _thread_model.reset(new DBServerThreadModel());
    _thread_model->set_server_proxy(_server_proxy);
    _db.reset(new DB());
}

DBServerController::~DBServerController() {

}

void DBServerController::initialize() {
    //register cmd handler
    _server_proxy->register_command_handler(COMMAND_ID_FE_OPERATION, 
        std::shared_ptr<CmdHandlerDBOperating>(new CmdHandlerDBOperating(shared_from_this())));
    //connect db
    std::string ip_port,user,pwd,db_name;
    AppConfig::instance()->get_db_info(ip_port, user, pwd, db_name);
    if(0 != _db->connect(user, ip_port, pwd, db_name) ) {
        MI_DBSERVER_LOG(MI_FATAL) << "connect to db failed.";
        DBSERVER_THROW_EXCEPTION("connect to db failed.");
    }
}

void DBServerController::run() {
    _thread_model->start();
    _server_proxy->run();
}

void DBServerController::finalize() {
    _thread_model->stop();
    _db->disconnect();
}

std::shared_ptr<DBServerThreadModel> DBServerController::get_thread_model() {
    return _thread_model;
}

std::shared_ptr<IPCServerProxy> DBServerController::get_server_proxy() {
    return _server_proxy;
}

std::shared_ptr<DB> DBServerController::get_db() {
    return _db;
}

MED_IMG_END_NAMESPACE