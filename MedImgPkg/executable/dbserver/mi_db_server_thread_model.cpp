#include "mi_db_server_thread_model.h"
#include "appcommon/mi_operation_interface.h"
#include "util/mi_ipc_server_proxy.h"
#include "mi_db_operation.h"


MED_IMG_BEGIN_NAMESPACE

DBServerThreadModel::DBServerThreadModel() {
    
}

DBServerThreadModel::~DBServerThreadModel() {

}

void DBServerThreadModel::set_server_proxy(std::shared_ptr<IPCServerProxy> proxy) {
    _server_proxy = proxy;
}

void DBServerThreadModel::push_operation(const std::shared_ptr<DBOperation>& op) {
    _op_msg_queue.push(op);
}

void DBServerThreadModel::start() {
    _op_msg_queue.activate();
    _thread_operating = boost::thread(boost::bind(&DBServerThreadModel::process_operating_i, this));
    _thread_sending = boost::thread(boost::bind(&DBServerThreadModel::process_sending_i, this));
}

void DBServerThreadModel::stop() {
    _op_msg_queue.deactivate();
    _thread_sending.interrupt();
    _thread_sending.join();
    _thread_operating.interrupt();
    _thread_operating.join();
}

void DBServerThreadModel::process_sending_i() {
    while(true) {
        _server_proxy->send();

        boost::this_thread::interruption_point();
    }
}

void DBServerThreadModel::process_operating_i() {

    while(true) {
        std::shared_ptr<DBOperation> op;
        _op_msg_queue.pop(&op);

        try {
            op->execute();
        } catch(const Exception& e) {
            MI_DBSERVER_LOG(MI_ERROR) << "op execute failed.";
        }

        boost::this_thread::interruption_point();
    }
}

MED_IMG_END_NAMESPACE