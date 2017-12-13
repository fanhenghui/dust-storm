#include "mi_ai_server_thread_model.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_operation_interface.h"
#include "util/mi_memory_shield.h"

#include "mi_ai_server_logger.h"

MED_IMG_BEGIN_NAMESPACE

AIServerThreadModel::AIServerThreadModel() {

}

AIServerThreadModel::~AIServerThreadModel() {

}

void AIServerThreadModel::set_client_proxy(std::shared_ptr<IPCClientProxy> proxy) {
    _client_proxy = proxy;
}

void AIServerThreadModel::push_operation(const std::shared_ptr<IOperation> op) {
    _op_msg_queue.push(op);
}

void AIServerThreadModel::async_send_data(IPCPackage* packages) {
    _pkg_msg_queue.push(packages);
}

void AIServerThreadModel::run() {
    try {
        _op_msg_queue.activate();
        _pkg_msg_queue.activate();
        _thread_operating = boost::thread(boost::bind(&AIServerThreadModel::process_operating, this));
        _thread_sending = boost::thread(boost::bind(&AIServerThreadModel::process_sending, this));
        _client_proxy->run();
    } catch (const Exception& e) {
        MI_AISERVER_LOG(MI_FATAL) << "AI client connect DB server thread exit with exception: " << e.what();
    } catch (const std::exception& e) {
        MI_AISERVER_LOG(MI_FATAL) << "AI client connect DB server thread exit with exception: " << e.what();
    } catch (...) {
        MI_AISERVER_LOG(MI_FATAL) << "AI client connect DB server thread exit with unexpected exception.";
    }   
    
}

void AIServerThreadModel::stop() {
    _thread_operating.interrupt();
    _thread_operating.join();
    _thread_sending.interrupt();
    _thread_sending.join();
    _client_proxy->stop();
    _op_msg_queue.deactivate();
}

void AIServerThreadModel::process_operating() {
    while(true) {
        std::shared_ptr<IOperation> op;
        _op_msg_queue.pop(&op);
        try {
            if (nullptr != op) {
                op->execute();
                op->reset();
            }
        }
        catch(const Exception& e) {
            MI_AISERVER_LOG(MI_ERROR) << "op execute failed with exception: " << e.what();
        }

        boost::this_thread::interruption_point();
    }
}

void AIServerThreadModel::process_sending() {
    while(true) {
        try {
            IPCPackage* pkg = nullptr;
            _pkg_msg_queue.pop(&pkg);
            if (nullptr != pkg) {
                StructShield<IPCPackage> shield(pkg);
                if (-1 == _client_proxy->sync_send_data(pkg->header, pkg->buffer) ) {
                    MI_AISERVER_LOG(MI_ERROR) << "send data failed;";    
                }
            }
        } catch(const Exception& e) {
            MI_AISERVER_LOG(MI_ERROR) << "send data failed with exception: " << e.what();
        }

        boost::this_thread::interruption_point();
    }
}

MED_IMG_END_NAMESPACE