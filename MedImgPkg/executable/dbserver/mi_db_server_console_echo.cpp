#include "mi_db_server_console_echo.h"

#include "util/mi_console_echo.h"
#include "util/mi_ipc_server_proxy.h"

#include "mi_db_server_logger.h"
#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

namespace {

class ActionDBS : public IEchoAction {
public:
    explicit ActionDBS(std::shared_ptr<DBServerController> controller):
    _controller(controller) {}

protected:
    std::weak_ptr<DBServerController> _controller;
private:
    DISALLOW_COPY_AND_ASSIGN(ActionDBS);
};

class ActionShutdown : public ActionDBS {
public:
    explicit ActionShutdown(std::shared_ptr<DBServerController> controller):ActionDBS(controller) {}
    virtual ~ActionShutdown() {};
    virtual int execute() {
        std::shared_ptr<DBServerController> controller = _controller.lock();
        if (controller) {
            controller->stop();
        }
        return ConsoleEcho::STOP_ECHO_SIGNAL;
    }
};

class ActionDBSBEStatus : public ActionDBS {
public:
    explicit ActionDBSBEStatus(std::shared_ptr<DBServerController> controller):ActionDBS(controller) {}
    virtual ~ActionDBSBEStatus() {};
    virtual int execute() {
        std::shared_ptr<DBServerController> controller = _controller.lock();
        if (controller) {
            std::shared_ptr<IPCServerProxy> proxy_be = controller->get_server_proxy_be();
            ServerStatus status = proxy_be->get_current_status();
            std::stringstream ss_client;
            for (auto it = status.client_hosts.begin(); it!= status.client_hosts.end(); ++it) {
                unsigned int socket_uid = it->first;
                size_t pkg_size = 0;
                auto it2 = status.client_packages.find(socket_uid);
                if(it2 != status.client_packages.end()) {
                    pkg_size = it2->second;
                }
                ss_client << "\{client host: " << it->second << ", pkg size: " << pkg_size << "}";
            }
            MI_DBSERVER_LOG(MI_INFO) << "\n{server host:" << status.host <<
            ", type:" << status.socket_type << ", client num:" << status.cur_client << 
            ", pkg capcity:" << status.package_cache_capcity << ", pkg size:" << status.package_cache_size << "}" 
            << ss_client.str();
        }
        return 0;
    }
};

class ActionDBSAIStatus : public ActionDBS {
public:
    explicit ActionDBSAIStatus(std::shared_ptr<DBServerController> controller):ActionDBS(controller) {}
    virtual ~ActionDBSAIStatus() {};
    virtual int execute() {
        std::shared_ptr<DBServerController> controller = _controller.lock();
        if (controller) {
            std::shared_ptr<IPCServerProxy> proxy_ais = controller->get_server_proxy_ais();
            ServerStatus status = proxy_ais->get_current_status();
            MI_DBSERVER_LOG(MI_INFO) << "{server host:" << status.host <<
            ", type:" << status.socket_type << ", client num:" << status.cur_client << 
            ", pkg capcity:" << status.package_cache_capcity << ", pkg size:" << status.package_cache_size; 
        }
        return 0;
    }
};



}

DBServerConsoleEcho::DBServerConsoleEcho(std::shared_ptr<DBServerController> controller): 
_controller(controller),_console_echo(new ConsoleEcho()) {
    init();
}

DBServerConsoleEcho::~DBServerConsoleEcho() {

}

void DBServerConsoleEcho::init() {
    //register echo actions
    std::shared_ptr<DBServerController> controller = _controller.lock();
    _console_echo->register_action("close", std::shared_ptr<ActionShutdown>(new ActionShutdown(controller)));
    _console_echo->register_action("bestatus", std::shared_ptr<ActionDBSBEStatus>(new ActionDBSBEStatus(controller)));
    _console_echo->register_action("aistatus", std::shared_ptr<ActionDBSAIStatus>(new ActionDBSAIStatus(controller)));
}

void DBServerConsoleEcho::run() {
    _console_echo->run();
}


MED_IMG_END_NAMESPACE