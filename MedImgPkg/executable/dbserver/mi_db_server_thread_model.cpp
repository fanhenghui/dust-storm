#include "mi_db_server_thread_model.h"

#include "util/mi_operation_interface.h"
#include "util/mi_ipc_server_proxy.h"
#include "util/mi_operation_interface.h"

#include "mi_db_server_console_echo.h"

MED_IMG_BEGIN_NAMESPACE

DBServerThreadModel::DBServerThreadModel() {
    
}

DBServerThreadModel::~DBServerThreadModel() {
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

void DBServerThreadModel::run() {
    try {
        _op_msg_queue_be.activate();
        _thread_be_operating = boost::thread(boost::bind(&DBServerThreadModel::process_be_operating_i, this));
        _thread_be_sending = boost::thread(boost::bind(&DBServerThreadModel::process_be_sending_i, this));
        _thread_be_recving = boost::thread(boost::bind(&DBServerThreadModel::process_be_recving_i, this));

        _op_msg_queue_ais.activate();
        _thread_ais_run = boost::thread(boost::bind(&DBServerThreadModel::process_ais_run_i, this));
        _thread_ais_operating = boost::thread(boost::bind(&DBServerThreadModel::process_ais_operating_i, this));
        _thread_ais_sending = boost::thread(boost::bind(&DBServerThreadModel::process_ais_sending_i, this));
        _thread_ais_recving = boost::thread(boost::bind(&DBServerThreadModel::process_ais_recving_i, this));

        boost::thread th_echo = boost::thread(boost::bind(&DBServerThreadModel::process_console_echo, this));
        th_echo.detach();

    
        _server_proxy_be->run();//main thread is accept APPS's client
    } catch (const Exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server accept BE clients exit with exception: " << e.what();
    } catch (const std::exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server accept BE clients exit with exception: " << e.what();
    } catch (...) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server accept BE clients exit with unexpected exception.";
    }    
}

void DBServerThreadModel::stop() {
    //stop can be call more than once

    _server_proxy_ais->stop();
    _server_proxy_be->stop();

    if (_thread_be_sending.joinable()) {
        _thread_be_sending.interrupt();
        _thread_be_sending.join();
    }
    
    if (_thread_be_operating.joinable()) {
        _thread_be_operating.interrupt();
        _thread_be_operating.join();
    }

    if (_thread_be_recving.joinable()) {
        _thread_be_recving.interrupt();
        _thread_be_recving.join();
    }

    if (_thread_ais_sending.joinable()) {
        _thread_ais_sending.interrupt();
        _thread_ais_sending.join();
    }

    if (_thread_ais_operating.joinable()) {
        _thread_ais_operating.interrupt();
        _thread_ais_operating.join();
    }

    if (_thread_ais_recving.joinable()) {
        _thread_ais_recving.interrupt();
        _thread_ais_recving.join();
    }

    if (_thread_ais_run.joinable()) {
        _thread_ais_run.join();
    }    

    _op_msg_queue_ais.deactivate();
    _op_msg_queue_be.deactivate();
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
            if (nullptr != op) {
                op->execute();
                op->reset();//release ipc data
            }
            
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
    try {
        _server_proxy_ais->run();
    } catch (const Exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server accept AI clients exit with exception: " << e.what();
        _server_proxy_be->stop();//stop main thread(blocking) then stop all
    } catch (const std::exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server accept AI clients exit with exception: " << e.what();
        _server_proxy_be->stop();//stop main thread(blocking) then stop all
    } catch (...) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server accept AI clients exit with unexpected exception.";
        _server_proxy_be->stop();//stop main thread(blocking) then stop all
    }    
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
            if (nullptr != op) {
                op->execute();
                op->reset();//release ipc data
            }
        } catch(const Exception& e) {
            MI_DBSERVER_LOG(MI_ERROR) << "op execute failed.";
        }

        boost::this_thread::interruption_point();
    }
}

MED_IMG_END_NAMESPACE