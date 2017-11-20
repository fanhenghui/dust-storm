#ifndef MEDIMG_MI_DB_SERVER_THREAD_MODEL_H
#define MEDIMG_MI_DB_SERVER_THREAD_MODEL_H

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

#include "util/mi_message_queue.h"

#include "mi_db_server_logger.h"
#include "mi_db_server_common.h"


MED_IMG_BEGIN_NAMESPACE

class IPCServerProxy;
class DBServerController;
class DBOperation;
class DBServerThreadModel {
public:
    DBServerThreadModel();
    ~DBServerThreadModel();

    void set_server_proxy(std::shared_ptr<IPCServerProxy> proxy);

    void push_operation(const std::shared_ptr<DBOperation>& op);

    void start();
    void stop();

private:
    void process_sending_i();
    void process_recving_i();
    void process_operating_i();
    void process_in_i();

private:
    std::weak_ptr<DBServerController> _controller;
    std::shared_ptr<IPCServerProxy> _server_proxy;

    MessageQueue<std::shared_ptr<DBOperation>> _op_msg_queue;

    boost::thread _thread_sending;
    boost::thread _thread_recving;
    boost::thread _thread_operating;    
    boost::thread _thread_in;    
};

MED_IMG_END_NAMESPACE


#endif