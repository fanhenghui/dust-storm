#include "mi_db_server_thread_model.h"
#include "appcommon/mi_operation_interface.h"
#include "util/mi_ipc_server_proxy.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

DBServerThreadModel::DBServerThreadModel() {
    
}

DBServerThreadModel::~DBServerThreadModel() {
    _thread_in.join();
}

void DBServerThreadModel::set_server_proxy_be(std::shared_ptr<IPCServerProxy> proxy) {
    _server_proxy_be = proxy;
}

void DBServerThreadModel::set_server_proxy_ais(std::shared_ptr<IPCServerProxy> proxy) {
    _server_proxy_ais = proxy;
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

    _thread_in = boost::thread(boost::bind(&DBServerThreadModel::process_in_i, this));

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

void DBServerThreadModel::process_in_i() {
    std::string msg;
    while(std::cin >> msg) {
        if(msg == "shutdown") {
            //quit signal
            _server_proxy_be->stop();
            _server_proxy_ais->stop();
            this->stop();
            break;
        } else if (msg == "status") {
            //server current status
            ServerStatus status = _server_proxy_be->get_current_status();
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
        } else if(msg == "aistatus") {
            //server current status
            ServerStatus status = _server_proxy_ais->get_current_status();
            MI_DBSERVER_LOG(MI_INFO) << "{server host:" << status.host <<
            ", type:" << status.socket_type << ", client num:" << status.cur_client << 
            ", pkg capcity:" << status.package_cache_capcity << ", pkg size:" << status.package_cache_size; 
        } else {
            MI_DBSERVER_LOG(MI_DEBUG) << "invalid msg."; 
        }
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