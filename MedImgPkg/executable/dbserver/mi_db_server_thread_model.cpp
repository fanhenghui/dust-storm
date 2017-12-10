#include "mi_db_server_thread_model.h"

#include "util/mi_operation_interface.h"
#include "util/mi_ipc_server_proxy.h"
#include "util/mi_operation_interface.h"

#include "mi_db_server_console_echo.h"

MED_IMG_BEGIN_NAMESPACE

DBServerThreadModel::DBServerThreadModel() {
    
}

DBServerThreadModel::~DBServerThreadModel() {
    _thread_console_echo.join();
}

void DBServerThreadModel::set_server_proxy_be(std::shared_ptr<IPCServerProxy> proxy) {
    _server_proxy_be = proxy;
}

void DBServerThreadModel::set_server_proxy_ais(std::shared_ptr<IPCServerProxy> proxy) {
    _server_proxy_ais = proxy;
}

void DBServerThreadModel::set_console_echo(std::shared_ptr<DBServerConsoleEcho> console_echo) {
    _console_echo = console_echo;
}

void DBServerThreadModel::push_operation_be(const std::shared_ptr<IOperation>& op) {
    _op_msg_queue_be.push(op);
}

void DBServerThreadModel::push_operation_ais(const std::shared_ptr<IOperation>& op) {
    _op_msg_queue_ais.push(op);
}

void DBServerThreadModel::start() {
    _op_msg_queue_be.activate();
    _thread_be_operating = boost::thread(boost::bind(&DBServerThreadModel::process_be_operating_i, this));
    _thread_be_sending = boost::thread(boost::bind(&DBServerThreadModel::process_be_sending_i, this));
    _thread_be_recving = boost::thread(boost::bind(&DBServerThreadModel::process_be_recving_i, this));

    _op_msg_queue_ais.activate();
    _thread_ais_run = boost::thread(boost::bind(&DBServerThreadModel::process_ais_run_i, this));
    _thread_ais_operating = boost::thread(boost::bind(&DBServerThreadModel::process_ais_operating_i, this));
    _thread_ais_sending = boost::thread(boost::bind(&DBServerThreadModel::process_ais_sending_i, this));
    _thread_ais_recving = boost::thread(boost::bind(&DBServerThreadModel::process_ais_recving_i, this));

    _thread_console_echo = boost::thread(boost::bind(&DBServerThreadModel::process_console_echo, this));

    _server_proxy_be->run();//main thread is accept APPS's client
}

void DBServerThreadModel::stop() {
    _op_msg_queue_be.deactivate();
    _thread_be_sending.interrupt();
    _thread_be_sending.join();
    _thread_be_operating.interrupt();
    _thread_be_operating.join();
    _thread_be_recving.interrupt();
    _thread_be_recving.join();

    _thread_ais_sending.interrupt();
    _thread_ais_sending.join();
    _thread_ais_operating.interrupt();
    _thread_ais_operating.join();
    _thread_ais_recving.interrupt();
    _thread_ais_recving.join();
    _thread_ais_run.join();
}

void DBServerThreadModel::process_be_sending_i() {
    while(true) {
        _server_proxy_be->send();

        boost::this_thread::interruption_point();
    }
}

void DBServerThreadModel::process_be_recving_i() {
    while(true) {
        _server_proxy_be->recv();

        boost::this_thread::interruption_point();
    }
}

void DBServerThreadModel::process_be_operating_i() {
    while(true) {
        std::shared_ptr<IOperation> op;
        _op_msg_queue_be.pop(&op);

        try {
            op->execute();
            op->reset();//release ipc data
        } catch(const Exception& e) {
            MI_DBSERVER_LOG(MI_ERROR) << "op execute failed with exception: " << e.what();
        }

        boost::this_thread::interruption_point();
    }
}

void DBServerThreadModel::process_console_echo() {
    if (_console_echo) {
        _console_echo->run();
    }
}

void DBServerThreadModel::process_ais_run_i() {
    _server_proxy_ais->run();
}

void DBServerThreadModel::process_ais_sending_i() {
    while(true) {
        _server_proxy_ais->send();

        boost::this_thread::interruption_point();
    }
}

void DBServerThreadModel::process_ais_recving_i() {
    while(true) {
        _server_proxy_ais->recv();

        boost::this_thread::interruption_point();
    }
}

void DBServerThreadModel::process_ais_operating_i() {
    while(true) {
        std::shared_ptr<IOperation> op;
        _op_msg_queue_ais.pop(&op);

        try {
            op->execute();
            op->reset();//release ipc data
        } catch(const Exception& e) {
            MI_DBSERVER_LOG(MI_ERROR) << "op execute failed.";
        }

        boost::this_thread::interruption_point();
    }
}

MED_IMG_END_NAMESPACE