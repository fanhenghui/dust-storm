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
class IOperation;
class DBServerThreadModel {
public:
    DBServerThreadModel();
    ~DBServerThreadModel();

    void set_server_proxy_be(std::shared_ptr<IPCServerProxy> proxy);
    void push_operation_be(const std::shared_ptr<IOperation>& op);

    void set_server_proxy_ais(std::shared_ptr<IPCServerProxy> proxy);
    void push_operation_ais(const std::shared_ptr<IOperation>& op);

    void start();
    void stop();

private:
    //for User input
    void process_in_i();

    //for INET BE
    void process_be_sending_i();
    void process_be_recving_i();
    void process_be_operating_i();

    //for UNIX AIS
    void process_ais_run_i();
    void process_ais_sending_i();
    void process_ais_recving_i();
    void process_ais_operating_i();

private:
    boost::thread _thread_in;

    //for BE client
    std::shared_ptr<IPCServerProxy> _server_proxy_be;
    MessageQueue<std::shared_ptr<IOperation>> _op_msg_queue_be;

    boost::thread _thread_be_sending;
    boost::thread _thread_be_recving;
    boost::thread _thread_be_operating;        

    //for AIS
    std::shared_ptr<IPCServerProxy> _server_proxy_ais;
    MessageQueue<std::shared_ptr<IOperation>> _op_msg_queue_ais;
    boost::thread _thread_ais_sending;
    boost::thread _thread_ais_recving;
    boost::thread _thread_ais_operating;
    boost::thread _thread_ais_run; 

private: 
    DISALLOW_COPY_AND_ASSIGN(DBServerThreadModel);
};

MED_IMG_END_NAMESPACE


#endif