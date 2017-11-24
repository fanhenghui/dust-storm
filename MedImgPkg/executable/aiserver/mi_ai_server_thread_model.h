#ifndef MEDIMGAIS_MI_AI_SERVER_THREAD_MODEL_H
#define MEDIMGAIS_MI_AI_SERVER_THREAD_MODEL_H

#include "mi_ai_server_common.h"
#include "util/mi_message_queue.h"
#include "util/mi_ipc_common.h"
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

MED_IMG_BEGIN_NAMESPACE

class IPCClientProxy;
class IOperation;
class AIServerThreadModel {
public:
    AIServerThreadModel();
    ~AIServerThreadModel();

    void set_client_proxy(std::shared_ptr<IPCClientProxy> proxy);
    void push_operation(const std::shared_ptr<IOperation> op);
    void async_send_data(IPCPackage* packages);

    void start();
    void stop();
    
private:
    void process_sending();
    void process_operating();    

private:
    std::shared_ptr<IPCClientProxy> _client_proxy;
    boost::thread _thread_sending;
    boost::thread _thread_operating;

    //calculate thread
    //TODO 多少个计算线程取决于显卡数量，调度原则是根据每个计算线程的负载(queue size)
    MessageQueue<std::shared_ptr<IOperation>> _op_msg_queue;

    //send result back to DBS queue
    MessageQueue<IPCPackage*> _pkg_msg_queue;
private:
    DISALLOW_COPY_AND_ASSIGN(AIServerThreadModel);
};

MED_IMG_END_NAMESPACE

#endif